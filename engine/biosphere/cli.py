"""``biosphere`` command line: serve, initdb, stitch, doctor."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from biosphere.core.config import get_settings
from biosphere.core.logging import configure_logging

app = typer.Typer(
    name="biosphere",
    help="BioSphere AI engine — video to embeddable 360 tour.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


# --------------------------------------------------------------------------------------
# serve
# --------------------------------------------------------------------------------------


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "",
    port: Annotated[int, typer.Option(help="Bind port.")] = 0,
    reload: Annotated[bool, typer.Option(help="Auto-reload on source changes (dev only).")] = False,
    workers: Annotated[int, typer.Option(help="Uvicorn worker processes. Ignored with --reload.")] = 1,
) -> None:
    """Run the API with uvicorn."""
    import uvicorn

    settings = get_settings()
    configure_logging()
    resolved_host = host or settings.host
    resolved_port = port or settings.port
    dev = reload or settings.env == "development"

    console.print(
        Panel.fit(
            f"[bold]{settings.app_name}[/bold]\n"
            f"http://{resolved_host}:{resolved_port}{'  (reload)' if reload else ''}\n"
            f"media  {settings.media_root}\n"
            f"db     {settings.database_url}",
            title="serving",
            border_style="cyan",
        )
    )
    uvicorn.run(
        "biosphere.api.main:app",
        host=resolved_host,
        port=resolved_port,
        reload=reload,
        # Multiple processes would each get their own in-process job queue and their own SSE
        # subscribers, so progress would only reach the client that happened to hit the right
        # worker. One process until the queue moves to a broker.
        workers=1 if (reload or workers <= 1) else workers,
        log_level="debug" if dev else "info",
        access_log=False,
    )


# --------------------------------------------------------------------------------------
# initdb
# --------------------------------------------------------------------------------------


@app.command()
def initdb() -> None:
    """Create the database schema and the media directory tree."""
    from biosphere.core.db import init_db

    settings = get_settings()
    init_db()
    for path in (settings.uploads_dir, settings.properties_dir, settings.work_dir, settings.cache_dir):
        path.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]ok[/green] schema ready at [bold]{settings.database_url}[/bold]")


# --------------------------------------------------------------------------------------
# stitch
# --------------------------------------------------------------------------------------


class _RichProgress:
    """ProgressReporter that drives a rich bar; the CLI's answer to the SSE stream."""

    def __init__(self, progress: Progress, task_id: Any) -> None:
        self._progress = progress
        self._task = task_id
        self._stage = ""

    def stage(self, name: str, fraction: float, message: str = "") -> None:
        from biosphere.core.jobs import STAGE_CUMULATIVE, STAGE_WEIGHTS

        overall = STAGE_CUMULATIVE.get(name, 0.0) + STAGE_WEIGHTS.get(name, 0.0) * max(
            0.0, min(fraction, 1.0)
        )
        self._stage = name
        self._progress.update(
            self._task, completed=overall * 100.0, description=f"[cyan]{name:<9}[/cyan]{message}"
        )

    def log(self, message: str, **fields: Any) -> None:
        console.log(f"[dim]{self._stage}[/dim] {message}", *([fields] if fields else []))


@app.command()
def stitch(
    video: Annotated[Path, typer.Argument(help="Capture to stitch (video file or image directory).")],
    mode: Annotated[str, typer.Option(help="fast | balanced | studio")] = "balanced",
    out: Annotated[Path, typer.Option(help="Output panorama path (.jpg).")] = Path("panorama.jpg"),
    work: Annotated[Path | None, typer.Option(help="Scratch directory. Defaults under the media root.")] = None,
) -> None:
    """Run the panorama pipeline on a local capture, outside the API."""
    from biosphere.core.storage import write_jpeg
    from biosphere.pano.types import PipelineConfig, QualityMode

    if not video.exists():
        console.print(f"[red]not found:[/red] {video}")
        raise typer.Exit(code=2)

    settings = get_settings()
    work_dir = work or (settings.work_dir / f"cli_{int(time.time())}")
    work_dir.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig.for_mode(QualityMode(mode))

    sources: Path | list[Path] = video
    if video.is_dir():
        images = sorted(
            p for p in video.iterdir() if p.suffix.lower() in settings.allowed_image_suffixes
        )
        if not images:
            console.print(f"[red]no images in[/red] {video}")
            raise typer.Exit(code=2)
        sources = images

    try:
        # Imported here so that `biosphere --help` never pays for the numeric stack.
        from biosphere.pano.pipeline import stitch_capture
    except ImportError as exc:
        console.print(f"[red]pipeline unavailable:[/red] {exc}")
        raise typer.Exit(code=3) from exc

    started = time.perf_counter()
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=32),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]starting[/cyan]", total=100.0)
        result = stitch_capture(
            sources, config, work_dir=work_dir, progress=_RichProgress(progress, task)
        )
        progress.update(task, completed=100.0, description="[green]done[/green]")

    out.parent.mkdir(parents=True, exist_ok=True)
    write_jpeg(out, result.panorama, config.jpeg_quality)

    table = Table(title="stitch metrics", header_style="bold cyan")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, value in result.metrics.as_dict().items():
        if isinstance(value, dict):
            value = ", ".join(f"{k}={v}" for k, v in value.items()) or "-"
        elif isinstance(value, list):
            value = ", ".join(str(v) for v in value) or "-"
        table.add_row(str(key), str(value))
    table.add_row("wall_clock_s", f"{time.perf_counter() - started:.1f}")
    table.add_row("output", str(out))
    console.print(table)

    coverage = result.coverage
    if coverage is not None and coverage.covered_fraction < 0.9:
        console.print(
            f"[yellow]warning:[/yellow] only {coverage.covered_fraction * 100:.0f}% of the sphere "
            "was covered by the capture."
        )


# --------------------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------------------


def _check_ffmpeg(binary: str) -> tuple[bool, str]:
    path = shutil.which(binary)
    if path is None:
        return False, "not on PATH"
    try:
        proc = subprocess.run([binary, "-version"], capture_output=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    first = (proc.stdout or b"").decode("utf-8", "replace").splitlines()
    return proc.returncode == 0, first[0] if first else path


@app.command()
def doctor() -> None:
    """Check every external dependency the engine relies on."""
    settings = get_settings()
    table = Table(title="biosphere doctor", header_style="bold cyan")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail", overflow="fold")

    def row(name: str, ok: bool, detail: str) -> None:
        table.add_row(name, "[green]ok[/green]" if ok else "[red]fail[/red]", detail)

    ok_ffmpeg, detail = _check_ffmpeg(settings.ffmpeg_bin)
    row("ffmpeg", ok_ffmpeg, detail)
    ok_ffprobe, detail = _check_ffmpeg(settings.ffprobe_bin)
    row("ffprobe", ok_ffprobe, detail)

    try:
        import cv2

        has_xphoto = hasattr(cv2, "xphoto")
        row("opencv", True, f"{cv2.__version__} (xphoto: {'yes' if has_xphoto else 'no'})")
    except Exception as exc:  # noqa: BLE001
        row("opencv", False, str(exc))

    try:
        import torch

        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else "cpu only"
        row("torch", True, f"{torch.__version__} — {name}")
    except Exception as exc:  # noqa: BLE001
        row("torch", False, f"optional; {exc}")

    media_ok = True
    detail = str(settings.media_root)
    try:
        settings.media_root.mkdir(parents=True, exist_ok=True)
        probe = settings.media_root / ".write_probe"
        probe.write_bytes(b"ok")
        probe.unlink()
    except OSError as exc:
        media_ok = False
        detail = f"{settings.media_root} — {exc}"
    row("media root writable", media_ok, detail)

    free_gb = shutil.disk_usage(settings.media_root).free / (1024**3) if media_ok else 0.0
    row("free space", free_gb >= 5.0, f"{free_gb:.1f} GB free")

    try:
        from biosphere.core.db import get_engine

        with get_engine().connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        row("database", True, settings.database_url)
    except Exception as exc:  # noqa: BLE001
        row("database", False, str(exc))

    try:
        import biosphere.pano.pipeline  # noqa: F401

        row("pipeline", True, "biosphere.pano.pipeline importable")
    except Exception as exc:  # noqa: BLE001
        row("pipeline", False, f"{type(exc).__name__}: {exc}")

    console.print(table)


# --------------------------------------------------------------------------------------
# user
# --------------------------------------------------------------------------------------

users = typer.Typer(name="user", help="Accounts.", no_args_is_help=True)
app.add_typer(users)


@users.command("create")
def user_create(
    email: Annotated[str, typer.Option(prompt=True, help="Sign-in address.")],
    name: Annotated[str, typer.Option(prompt=True, help="Shown in the dashboard.")],
    password: Annotated[
        str, typer.Option(prompt=True, confirmation_prompt=True, hide_input=True, help="At least 10 characters.")
    ],
) -> None:
    """Create an account from the server.

    The way to onboard somebody on an installation where ``BIOSPHERE_ALLOW_REGISTRATION`` is
    off — and the way to make the FIRST account, which is also the one that adopts any
    properties created before this engine had logins.
    """
    from sqlmodel import select

    from biosphere.core import auth as auth_lib
    from biosphere.core.db import init_db, session_scope
    from biosphere.core.models import Property, User

    init_db()
    address = auth_lib.normalise_email(email)
    try:
        auth_lib.check_password_policy(password)
    except auth_lib.PasswordPolicyError as exc:
        console.print(f"[red]{exc.message_en}[/red]")
        raise typer.Exit(code=1) from exc

    with session_scope() as session:
        if session.exec(select(User).where(User.email == address)).first() is not None:
            console.print(f"[red]An account already exists for {address}.[/red]")
            raise typer.Exit(code=1)
        first = session.exec(select(User)).first() is None
        user = User(email=address, name=name.strip(), password_hash=auth_lib.hash_password(password))
        session.add(user)
        session.flush()
        adopted = 0
        if first:
            for prop in session.exec(select(Property).where(Property.owner_id.is_(None))).all():  # type: ignore[union-attr]
                prop.owner_id = user.id
                session.add(prop)
                adopted += 1

    console.print(f"[green]ok[/green] account created for [bold]{address}[/bold]")
    if adopted:
        console.print(f"     adopted {adopted} propert{'y' if adopted == 1 else 'ies'} that had no owner")


@users.command("list")
def user_list() -> None:
    """Every account, with how many properties it owns."""
    from sqlmodel import func, select

    from biosphere.core.db import init_db, session_scope
    from biosphere.core.models import Property, User

    init_db()
    table = Table("e-mail", "name", "sign-in", "properties", "state", title="Accounts")
    with session_scope() as session:
        for user in session.exec(select(User).order_by(User.created_at)).all():  # type: ignore[arg-type]
            owned = int(
                session.exec(select(func.count()).select_from(Property).where(Property.owner_id == user.id)).one()
            )
            methods = ", ".join(
                filter(None, ["password" if user.password_hash else "", "google" if user.google_sub else ""])
            )
            table.add_row(
                user.email, user.name, methods or "—", str(owned), "disabled" if user.disabled else "active"
            )
    console.print(table)


@users.command("passwd")
def user_passwd(
    email: Annotated[str, typer.Option(prompt=True)],
    password: Annotated[str, typer.Option(prompt=True, confirmation_prompt=True, hide_input=True)],
) -> None:
    """Set a new password, and sign every one of that account's browsers out.

    Revoking the sessions is the point rather than a side effect: a password is changed
    because it might be known, and leaving the existing sessions alive would leave whoever
    knew it signed in.
    """
    from sqlmodel import select

    from biosphere.core import auth as auth_lib
    from biosphere.core.db import init_db, session_scope
    from biosphere.core.models import AuthSession, User

    init_db()
    address = auth_lib.normalise_email(email)
    try:
        auth_lib.check_password_policy(password)
    except auth_lib.PasswordPolicyError as exc:
        console.print(f"[red]{exc.message_en}[/red]")
        raise typer.Exit(code=1) from exc

    with session_scope() as session:
        user = session.exec(select(User).where(User.email == address)).first()
        if user is None:
            console.print(f"[red]No account for {address}.[/red]")
            raise typer.Exit(code=1)
        user.password_hash = auth_lib.hash_password(password)
        session.add(user)
        revoked = 0
        for row in session.exec(select(AuthSession).where(AuthSession.user_id == user.id)).all():
            session.delete(row)
            revoked += 1

    console.print(f"[green]ok[/green] password set for [bold]{address}[/bold]; {revoked} session(s) signed out")


@users.command("disable")
def user_disable(
    email: Annotated[str, typer.Option(prompt=True)],
    enable: Annotated[bool, typer.Option("--enable", help="Turn the account back on instead.")] = False,
) -> None:
    """Lock an account out without deleting anything it owns."""
    from sqlmodel import select

    from biosphere.core import auth as auth_lib
    from biosphere.core.db import init_db, session_scope
    from biosphere.core.models import AuthSession, User

    init_db()
    address = auth_lib.normalise_email(email)
    with session_scope() as session:
        user = session.exec(select(User).where(User.email == address)).first()
        if user is None:
            console.print(f"[red]No account for {address}.[/red]")
            raise typer.Exit(code=1)
        user.disabled = not enable
        session.add(user)
        if user.disabled:
            for row in session.exec(select(AuthSession).where(AuthSession.user_id == user.id)).all():
                session.delete(row)
    console.print(f"[green]ok[/green] {address} is now {'active' if enable else 'disabled'}")


# At the very end, not after the first group of commands: ``python -m biosphere.cli`` runs this
# module top to bottom, and a ``user`` group registered below an early ``app()`` call did not
# exist yet when Typer parsed the command line ("No such command 'user'"). The installed
# ``biosphere`` entry point imports the whole module first, which is why only it worked.
if __name__ == "__main__":  # pragma: no cover
    app()
