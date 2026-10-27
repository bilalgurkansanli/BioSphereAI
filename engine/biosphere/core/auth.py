"""Passwords and sessions.

Three decisions here are worth stating, because each one is a place where the obvious
implementation is wrong in a way that never shows up in testing.

*The password hash is scrypt from the standard library, not a new dependency.* A KDF is one
of the few pieces of security code where "well-reviewed and already installed" beats
"popular": ``hashlib.scrypt`` is OpenSSL's implementation, it is memory-hard, and it costs
this project no supply chain. The parameters below are written into every stored hash, so
raising them later re-hashes people on their next successful login instead of locking them
out — see :func:`needs_rehash`.

*The session token is stored as a SHA-256 digest, never as itself.* A session token is a
bearer credential: whoever holds it is the user. A database dump that contains live tokens
is a database dump that logs the attacker in, and unlike passwords there is nothing the
victim can change. The digest is enough to look a session up, because we are handed the
token on every request and can hash it again. SHA-256 with no salt and no stretching is the
right primitive here and a slow KDF would be the wrong one: the token already has 256 bits
of entropy from ``secrets``, so there is nothing to brute-force and nothing to make slower.

*Login does the same work whether or not the account exists.* Skipping the hash for an
unknown e-mail turns the response time into an oracle that answers "is this person a
customer?" for anyone with a list of addresses — which, for a product whose customers are
estate agencies, is a competitor's prospect list. :func:`verify_password` is therefore
called against a dummy hash when no user is found.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta

__all__ = [
    "hash_password",
    "verify_password",
    "needs_rehash",
    "new_session_token",
    "token_digest",
    "SESSION_TTL",
    "SESSION_TOKEN_BYTES",
    "PasswordPolicyError",
    "check_password_policy",
    "normalise_email",
]

# scrypt parameters. n is the cost, and the memory used is roughly ``128 * n * r`` bytes —
# 32 MiB here. That is deliberately more than a web request usually allocates: it is the
# whole point of a memory-hard KDF, and it is what makes a GPU a poor attacker. On this
# project's hardware one hash takes on the order of 60 ms, which a person cannot feel at a
# login screen and an attacker feels a great deal.
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SALT_BYTES = 16

#: Long enough that guessing is hopeless, short enough to sit in a header.
SESSION_TOKEN_BYTES = 32

#: How long a session stays valid without being refreshed.
SESSION_TTL = timedelta(days=30)

#: Cost of a single scrypt call, paid on a login for an address that has no account so the
#: response time says nothing about whether it does. Computed once, at import.
_DUMMY_HASH: str | None = None


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


class PasswordPolicyError(ValueError):
    """A password the product refuses to store. Carries copy in both languages."""

    def __init__(self, message_en: str, message_tr: str) -> None:
        super().__init__(message_en)
        self.message_en = message_en
        self.message_tr = message_tr


def check_password_policy(password: str) -> None:
    """Length only, on purpose.

    Composition rules ("one uppercase, one digit, one symbol") measurably push people toward
    ``Password1!`` and its neighbours, which is a smaller search space than a long passphrase.
    Length is the requirement that actually correlates with strength, so it is the only one
    enforced — plus an upper bound, because scrypt happily hashes a 10 MB "password" and that
    is a denial-of-service primitive rather than a security feature.
    """
    if len(password) < 10:
        raise PasswordPolicyError(
            "The password must be at least 10 characters long.",
            "Şifre en az 10 karakter olmalı.",
        )
    if len(password.encode("utf-8")) > 1024:
        raise PasswordPolicyError(
            "The password is too long.",
            "Şifre çok uzun.",
        )


def hash_password(password: str) -> str:
    """``scrypt$n$r$p$salt$hash``, everything needed to verify it and to know when to redo it."""
    salt = secrets.token_bytes(_SALT_BYTES)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
        maxmem=_SCRYPT_N * 2 * _SCRYPT_R * 64 + 1024 * 1024,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64(salt)}${_b64(derived)}"


def _dummy_hash() -> str:
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password(secrets.token_urlsafe(16))
    return _DUMMY_HASH


def verify_password(password: str, stored: str | None) -> bool:
    """Constant-time check. ``stored`` may be ``None`` for a Google-only account.

    Passing ``None`` still costs a full scrypt call: an account that exists but has no
    password must not answer faster than one whose password is simply wrong.
    """
    target = stored if stored else _dummy_hash()
    try:
        scheme, n, r, p, salt_b64, hash_b64 = target.split("$")
        if scheme != "scrypt":
            return False
        salt = _unb64(salt_b64)
        expected = _unb64(hash_b64)
        n_i, r_i, p_i = int(n), int(r), int(p)
    except (ValueError, TypeError):
        return False

    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n_i,
        r=r_i,
        p=p_i,
        dklen=len(expected),
        maxmem=n_i * 2 * r_i * 64 + 1024 * 1024,
    )
    matched = hmac.compare_digest(derived, expected)
    # A password-less account can never be signed into with a password, however well the
    # dummy comparison went.
    return matched and bool(stored)


def needs_rehash(stored: str | None) -> bool:
    """True when the stored hash was made with weaker parameters than we now use."""
    if not stored:
        return False
    try:
        scheme, n, r, p, _salt, _hash = stored.split("$")
    except ValueError:
        return True
    if scheme != "scrypt":
        return True
    return (int(n), int(r), int(p)) != (_SCRYPT_N, _SCRYPT_R, _SCRYPT_P)


def new_session_token() -> str:
    """256 bits, URL-safe. This is the whole credential; it is shown to the client once."""
    return secrets.token_urlsafe(SESSION_TOKEN_BYTES)


def token_digest(token: str) -> str:
    """What goes in the database in place of the token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def session_expiry(now: datetime | None = None) -> datetime:
    return (now or datetime.now(UTC)) + SESSION_TTL


#: How long a capture link (the QR code a phone scans) keeps working.
#:
#: A WORKING DAY, not a month like a session. The link travels as a picture — on a screen in
#: an office, in a photo of that screen, in a WhatsApp message to whoever is visiting the flat
#: — and a picture cannot be logged out. Twelve hours covers "made the QR in the morning,
#: filmed the flat in the afternoon"; anything longer is a credential lying around. A new QR
#: costs one click on the property page.
CAPTURE_GRANT_TTL = timedelta(hours=12)


def capture_grant_expiry(now: datetime | None = None) -> datetime:
    return (now or datetime.now(UTC)) + CAPTURE_GRANT_TTL


def normalise_email(email: str) -> str:
    """Lower-cased and stripped.

    The local part of an address is case-sensitive by RFC and case-insensitive at every mail
    provider anyone uses. Following the RFC here would let ``Ali@x.com`` and ``ali@x.com``
    become two accounts that the same person cannot tell apart — so the address is stored
    folded, and the uniqueness constraint means what a user expects it to mean.
    """
    return email.strip().lower()
