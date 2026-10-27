"""Runtime configuration, resolved once at import time.

Everything is overridable through environment variables prefixed with ``BIOSPHERE_``
or an ``.env`` file next to the ``engine`` package.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENGINE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ENGINE_ROOT.parent


def _default_media_root() -> Path:
    """Prefer a roomy volume: panoramas are large and the system drive is usually not."""
    override = os.environ.get("BIOSPHERE_MEDIA_ROOT")
    if override:
        return Path(override)
    for candidate in (Path("E:/biosphere-data"), Path("D:/biosphere-data")):
        if candidate.drive and Path(candidate.drive + "/").exists():
            return candidate
    return REPO_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BIOSPHERE_",
        env_file=(ENGINE_ROOT / ".env", REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- service ----------------------------------------------------------------------
    app_name: str = "BioSphere AI Engine"
    env: str = "development"
    host: str = "127.0.0.1"
    # 8000 is the single most contended port on a developer machine — this box already
    # runs another FastAPI service there. 8008 keeps `dev.ps1` from failing to bind on a
    # fresh clone for a reason that has nothing to do with this project.
    port: int = 8008
    public_base_url: str = "http://127.0.0.1:8008"
    web_base_url: str = "http://localhost:3000"

    # -- CORS -------------------------------------------------------------------------
    # The player must be embeddable on arbitrary listing portals, so the *public* read
    # routes and the media mount allow any origin. Write routes are restricted below.
    #
    # 3007 is here because `.claude/launch.json` has a `web-3007` configuration for exactly
    # the case where 3000 is taken by another project on the same machine. A dashboard that
    # cannot reach the engine fails as a wall of CORS errors in the console and an "engine
    # offline" pill — a diagnosis nobody enjoys twice. Both are loopback, so this widens
    # nothing an attacker can reach, and a deployment overrides the whole list anyway.
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3007",
            "http://127.0.0.1:3007",
        ]
    )
    cors_allow_all_public_reads: bool = True

    # -- storage ----------------------------------------------------------------------
    media_root: Path = Field(default_factory=_default_media_root)
    media_url_prefix: str = "/media"
    database_url: str = ""

    # -- upload limits ----------------------------------------------------------------
    #: Sized against the client's own capture ceiling, not picked round. `plan.MAX_CAPTURE_MS`
    #: is 600 s and `RECORDING.targetBitrateBps` is 12 Mbps, so the longest capture the phone
    #: will produce is about 900 MB before container overhead; 1.5 GB leaves room for that,
    #: for the `motion.json` sidecar, and for a capture split into several segments whose
    #: headers repeat. A limit under what the client can shoot is the worst kind: the user
    #: finds out only after filming the room and uploading all of it.
    max_upload_bytes: int = 1536 * 1024 * 1024
    min_video_seconds: float = 5.0
    max_video_seconds: float = 300.0
    allowed_video_suffixes: tuple[str, ...] = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".3gp")
    allowed_image_suffixes: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif")

    # -- processing -------------------------------------------------------------------
    worker_concurrency: int = 1
    default_quality_mode: str = "balanced"
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    keep_work_dirs: bool = False
    enable_gpu: bool = True

    # -- ai ---------------------------------------------------------------------------
    ai_models_dir: Path | None = None
    ai_allow_downloads: bool = False

    #: Where the GPU work happens. Empty means "in this process".
    #:
    #: THE MODELS DO NOT HAVE TO LIVE WHERE THE SERVICE DOES, and on a workstation that is
    #: also somebody's desktop they should not. Emptying a room is seconds of a whole GPU;
    #: this box runs the media library, the job queue and the person's other work, while the
    #: card that would do it well is in a different machine on the same desk. Point this at
    #: a `biosphere.ai.worker` running there and the heavy stage moves without anything else
    #: moving with it — the media root, the database and the queue all stay put.
    #:
    #: e.g. ``http://192.168.1.42:8010``
    ai_worker_url: str = ""

    #: Whether this process may load models onto its own GPU.
    #:
    #: Forced off whenever `ai_worker_url` is set: two copies of the same weights on two
    #: machines is not a fallback, it is the thing this setting exists to prevent.
    ai_local_models: bool = True

    #: Which inpainter fills the room: ``auto`` | ``sd`` | ``lama``.
    #:
    #: ``auto`` means LaMa when its weights are present and the classical per-surface fill
    #: when they are not. IT DELIBERATELY NEVER SELECTS THE DIFFUSION PATH: on a real
    #: photographed bedroom that model draws a new bed into the bed-shaped hole rather than
    #: removing the old one, and no prompt or guidance setting talks it out of that — a
    #: generative inpainter answers "what belongs here" and cannot be asked for an absence.
    #: Measured numbers are in :mod:`biosphere.ai.sd_inpaint`.
    #:
    #: ``sd`` opts into it anyway, which is reasonable for small removals and for anyone
    #: wanting to keep a run off LaMa's non-commercial weights — but look at the result before
    #: putting it in front of a buyer. ``lama`` pins the reconstruction path and refuses to
    #: fall back.
    ai_inpainter: str = "auto"

    # -- listing copy -----------------------------------------------------------------
    #: Key for the hosted model that writes a property's title and description.
    #:
    #: A DIFFERENT KIND OF DEPENDENCY FROM EVERYTHING ABOVE IT. The inpainter and the
    #: segmenter are weights this project can put on a disk; this is somebody else's service,
    #: billed per call, and it is the only part of the engine that leaves the building. So it
    #: is off unless a key is set, and the feature it powers keeps a deterministic path that
    #: works without one — see :mod:`biosphere.ai.listing`.
    #:
    #: Left empty, ``ANTHROPIC_API_KEY`` from the environment is used, because that is the
    #: name every other tool on the machine already reads.
    listing_api_key: str = ""

    #: Which model writes it. Vision is required — the description is written from the rooms'
    #: own pictures, not only from their names.
    #:
    #: HAIKU BY DEFAULT, AND THAT IS A PRICE DECISION TAKEN DELIBERATELY. One listing on Opus 5
    #: measured 7-8 cents; on Haiku 4.5 the same work is about a twentieth of that, and this is
    #: a button an agent presses on every property they shoot. What the cheaper model gives up
    #: is prose quality, not truthfulness: the numbers still come from the FACTS block and
    #: `listing.audit` still checks every one of them, so a weaker writer produces a flatter
    #: paragraph rather than a wrong one. An agency that would rather pay for the better
    #: sentence sets `claude-sonnet-5` or `claude-opus-5` here.
    listing_model: str = "claude-haiku-4-5"

    #: How many room views travel with one request.
    #:
    #: Sized by what a description can honestly use rather than by what fits: six rooms is
    #: already more than any listing paragraph names, and every image is tokens on somebody's
    #: bill. The rooms are chosen in property order with the cover first, so the six are the six
    #: an agent put at the front.
    listing_max_images: int = 6

    #: Ceiling for one generation, including the upload of those images.
    listing_timeout_s: float = 180.0

    # -- investment assistant -----------------------------------------------------------
    #: The buyer-facing chat inside the player (``biosphere.invest``). Off per property until
    #: the agent opts in and enters a price; this switch turns it off for the whole
    #: installation, leaving only the deterministic figures card.
    advisor_enabled: bool = True

    #: Key for the assistant's model. Empty uses the listing key, which itself falls back to
    #: ``ANTHROPIC_API_KEY`` — one hosted-model account is the normal shape of a deployment.
    advisor_api_key: str = ""

    #: Which model answers. SONNET 5.5 BY DEFAULT — Bilal's choice (2026-10-07) on the bench
    #: below: it traced every number as Opus did, at 45 % less a question and faster. Measured on
    #: 10 buyer questions (``scripts/bench_advisor.py``, 2026-10-06, basic search): Sonnet 5.5 low
    #: $0.031 a question, median 7.6 s, no untraced number; Opus 5.5 low $0.056, median 10-13 s,
    #: none; Haiku 4.5 $0.015, 7.0 s, but it does arithmetic of its own (sums a fee and its KDV,
    #: works out 12 × rent) and leaves the calculators unused on comparisons — not recommended
    #: for this feature. `claude-opus-5-5` is one setting away when the answer matters more.
    advisor_model: str = "claude-sonnet-5-5"

    #: ``low`` | ``medium`` | ``high``. Set explicitly because Opus 5.5 defaults to medium and
    #: Sonnet 5.5 to high — leaving it out would make the two models' bills incomparable.
    #: LOW BY DEFAULT: on the bench, Opus 5.5 at medium cost 41 % more a question ($0.079 vs
    #: $0.056) and took longer (median 14.7 s vs 9.8-13.2 s) for answers that traced every number
    #: just as well. Ignored on Haiku 4.5, which takes no effort setting.
    advisor_effort: str = "low"

    #: Output ceiling of ONE call, thinking included (thinking counts against it and cannot be
    #: switched off on Opus 5.5). An answer is a few hundred tokens; the rest is headroom.
    advisor_max_output_tokens: int = 8000

    #: Calls one question may take: search and compute, compute again, answer. The last one is
    #: made with tools switched off, so a question always ends in an answer.
    advisor_max_rounds: int = 4

    #: Web searches allowed in one call (the tool's ``max_uses``). Fixed for a conversation —
    #: the tool list is part of the cached prefix — so the per-question ceiling is this times
    #: ``advisor_max_rounds``; the prompt asks for at most two.
    advisor_max_searches_per_call: int = 3

    #: Whether the model may search the web at all. Off, it answers from the agent's figures
    #: and the calculators only, and says that market data is unavailable.
    advisor_web_search: bool = True

    #: Which web search: ``basic`` (``web_search_20250305``: the results go straight into the
    #: context and the answer carries citations), ``dynamic`` (``web_search_20260209``: the
    #: model runs code over the results first) or ``auto`` (the newest the model takes).
    #:
    #: BASIC BY DEFAULT, measured: on the four search questions Opus 5.5 low took a median
    #: 17.7 s with basic search and 31.8 s with dynamic filtering, at about the same cost
    #: ($0.104 vs $0.114 a question). Dynamic filtering also returns the page text it read
    #: encrypted and attaches no citations, so neither the panel's source chips nor the
    #: number checker can see where a market figure came from.
    advisor_web_search_version: str = "basic"

    #: Restrict web search to these domains (e.g. ``["tcmb.gov.tr","tuik.gov.tr","endeksa.com"]``).
    #: Empty means unrestricted, which is what finds a district's rents in practice.
    advisor_web_allowed_domains: list[str] = Field(default_factory=list)

    #: Retry a classifier refusal on Anthropic's recommended fallback model (server side,
    #: ``fallbacks: "default"``). Only Opus 5.5 / Sonnet 5.5 accept it.
    advisor_fallbacks: bool = True

    #: Questions one conversation may ask, and the length of one question.
    advisor_max_turns: int = 12
    advisor_max_message_chars: int = 600

    #: A conversation forgets itself after this long without a question. Conversations live
    #: in memory only — never on disk — and at most ``advisor_max_conversations`` at once.
    advisor_conversation_ttl_minutes: int = 60
    advisor_max_conversations: int = 500

    #: A conversation whose last call read more than this many prompt tokens takes no more
    #: questions: every turn re-reads the whole history, so a long chat costs more per turn.
    advisor_max_context_tokens: int = 80000

    #: Daily spending caps in USD, on the Europe/Istanbul day. Per property (the agent may set
    #: a lower or higher one up to the max) and across the installation.
    advisor_daily_budget_usd: float = 2.0
    advisor_daily_budget_max_usd: float = 10.0
    advisor_global_daily_budget_usd: float = 25.0

    #: Questions per visitor IP and per property. Counted in memory against a salted digest
    #: of the address; nothing is written down.
    advisor_rate_ip_per_minute: int = 6
    advisor_rate_ip_per_hour: int = 40
    advisor_rate_property_per_hour: int = 150
    #: Read the client address from ``X-Forwarded-For``. Only behind a proxy you control —
    #: otherwise any visitor can pick a new "IP" per request and walk around the limit.
    advisor_trust_forwarded_for: bool = False

    #: Answers streaming at the same time, across the installation.
    advisor_max_concurrent: int = 8

    #: Ceiling for one call, including its web searches.
    advisor_timeout_s: float = 120.0

    #: Overrides for the Turkish cost parameters (tapu harcı, döner sermaye, commission, the
    #: example mortgage rate, BDDK's LTV bands...) as a JSON object keyed like
    #: ``biosphere.invest.rates.CostRates``. Empty uses the dated defaults in that module.
    advisor_rates: str = ""

    # -- accounts ---------------------------------------------------------------------
    #: Whether anybody may create an account, or only the operator (through the CLI).
    #:
    #: On by default so a fresh self-hosted install can be used at once: the first account
    #: created also adopts any data that predates accounts. Turn it off on a public instance.
    #: Turning it off does NOT disable Google sign-in for people who already have an account
    #: — only the creation of new ones, by either route.
    allow_registration: bool = True

    #: ``local`` | ``accounts``. How the dashboard is entered.
    #:
    #: LOCAL BY DEFAULT because BioSphere is open-source software people run on their own
    #: machine for themselves: a sign-in form in front of your own dashboard, on your own
    #: laptop, protects nothing and costs a step. In local mode a request without a session acts
    #: as the installation's owner — its oldest account, or one created on first use — while
    #: QR capture links, the public embed routes and real sessions work exactly as before.
    #: Running an instance OTHER people can reach? Set ``accounts``: every private route then
    #: needs a sign-in again. The engine binds to 127.0.0.1 by default, so local mode is not
    #: reachable from the network unless you change that.
    auth_mode: str = "local"

    #: Google sign-in. Both must be set for the button to appear; the login screen asks the
    #: engine (`/auth/google/status`) rather than guessing, so an installation without a
    #: Google project simply does not show it.
    #:
    #: Create them at console.cloud.google.com → APIs & Services → Credentials → OAuth
    #: client ID (Web application), and add the callback below to "Authorised redirect URIs"
    #: EXACTLY as it is sent, including the scheme and any port. Google compares that string
    #: character for character and its error for a mismatch names no fix.
    google_client_id: str = ""
    google_client_secret: str = ""

    #: Left empty, the callback is derived as ``{public_base_url}/api/v1/auth/google/callback``.
    #: Set it explicitly when the engine sits behind a proxy on another hostname, because
    #: what must match in Google's console is the URL THE BROWSER visits, not the one the
    #: engine believes it is serving.
    google_redirect_url: str = ""

    @property
    def ai_use_local_models(self) -> bool:
        """Local models only when nothing better is configured."""
        return bool(self.ai_local_models) and not self.ai_worker_url.strip()

    @property
    def listing_key(self) -> str:
        """The key actually used: the dashboard's, else this setting, else ``ANTHROPIC_API_KEY``."""
        from biosphere.core.api_keys import stored_api_key

        explicit = (self.listing_api_key or "").strip()
        return stored_api_key() or explicit or os.environ.get("ANTHROPIC_API_KEY", "").strip()

    @property
    def advisor_key(self) -> str:
        """The investment assistant's key: its own setting, else the listing key's chain."""
        from biosphere.core.api_keys import stored_api_key

        return stored_api_key() or (self.advisor_api_key or "").strip() or self.listing_key

    @field_validator("media_root", mode="after")
    @classmethod
    def _ensure_media_root(cls, v: Path) -> Path:
        v = Path(v).expanduser()
        for sub in ("uploads", "properties", "restyle", "work", "cache", "db"):
            (v / sub).mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("database_url", mode="after")
    @classmethod
    def _default_db(cls, v: str, info) -> str:
        if v:
            return v
        root = info.data.get("media_root")
        root = Path(root) if root else _default_media_root()
        return f"sqlite:///{(root / 'db' / 'biosphere.db').as_posix()}"

    # -- derived paths ----------------------------------------------------------------
    @property
    def uploads_dir(self) -> Path:
        return self.media_root / "uploads"

    @property
    def properties_dir(self) -> Path:
        return self.media_root / "properties"

    @property
    def restyle_dir(self) -> Path:
        """Delivered virtual-staging renders. Publicly mounted, like ``properties``.

        Its own tree rather than a folder inside ``properties`` because a restyle has no property: the
        feature takes a bare photograph, addresses it by the hash of its own bytes, and knows
        nothing about who owns it. Putting that under a property id would mean inventing one.
        """
        return self.media_root / "restyle"

    @property
    def work_dir(self) -> Path:
        return self.media_root / "work"

    @property
    def cache_dir(self) -> Path:
        return self.media_root / "cache"

    def media_url(self, path: Path) -> str:
        """Map an absolute path under ``media_root`` to its public URL."""
        rel = Path(path).resolve().relative_to(self.media_root.resolve()).as_posix()
        return f"{self.media_url_prefix}/{rel}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
