"""Accounts, sessions and ownership.

The client here is deliberately ANONYMOUS: ``test_api`` signs its own client in through an
autouse fixture that is not imported with the fixtures below, so every request in this file
starts with no credentials and says so.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from biosphere.core import auth as auth_lib
from biosphere.core.config import Settings
from biosphere.core.db import get_engine
from biosphere.core.models import AuthSession, Property, User

from .test_api import API, client, settings  # noqa: F401 - fixtures

GOOD_PASSWORD = "correct-horse-battery"


def register(test_client: TestClient, email: str = "ada@example.com", name: str = "Ada") -> dict:
    response = test_client.post(
        f"{API}/auth/register", json={"email": email, "name": name, "password": GOOD_PASSWORD}
    )
    assert response.status_code == 201, response.text
    return response.json()


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ======================================================================================
# Registration and sign-in
# ======================================================================================


def test_register_returns_a_token_that_identifies_the_account(client: TestClient) -> None:
    body = register(client)
    assert body["user"]["email"] == "ada@example.com"
    assert body["user"]["has_password"] is True
    assert body["user"]["has_google"] is False

    me = client.get(f"{API}/auth/me", headers=auth(body["token"]))
    assert me.status_code == 200
    assert me.json()["id"] == body["user"]["id"]


def test_the_email_is_folded_so_one_address_cannot_become_two_accounts(client: TestClient) -> None:
    register(client, email="Ada@Example.COM")
    again = client.post(
        f"{API}/auth/register",
        json={"email": "ada@example.com", "name": "Impostor", "password": GOOD_PASSWORD},
    )
    assert again.status_code == 409
    assert again.json()["code"] == "email_taken"

    signed_in = client.post(f"{API}/auth/login", json={"email": "  ADA@example.com ", "password": GOOD_PASSWORD})
    assert signed_in.status_code == 200


def test_a_short_password_is_refused_with_copy_in_both_languages(client: TestClient) -> None:
    response = client.post(
        f"{API}/auth/register", json={"email": "b@example.com", "name": "B", "password": "kısa"}
    )
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "weak_password"
    assert body["message_tr"] and body["message_tr"] != body["message"]


def test_a_wrong_password_and_an_unknown_address_are_indistinguishable(client: TestClient) -> None:
    register(client)
    wrong = client.post(f"{API}/auth/login", json={"email": "ada@example.com", "password": "not-the-password"})
    unknown = client.post(f"{API}/auth/login", json={"email": "nobody@example.com", "password": GOOD_PASSWORD})

    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_logout_revokes_the_token_it_was_called_with(client: TestClient) -> None:
    token = register(client)["token"]
    assert client.post(f"{API}/auth/logout", headers=auth(token)).status_code == 204
    assert client.get(f"{API}/auth/me", headers=auth(token)).status_code == 401


def test_an_expired_session_is_rejected_and_deleted(client: TestClient, settings: Settings) -> None:
    token = register(client)["token"]
    with Session(get_engine()) as session:
        row = session.exec(select(AuthSession)).one()
        row.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        session.add(row)
        session.commit()

    assert client.get(f"{API}/auth/me", headers=auth(token)).status_code == 401
    with Session(get_engine()) as session:
        assert session.exec(select(AuthSession)).all() == []


def test_a_disabled_account_cannot_sign_in_or_use_a_live_session(client: TestClient) -> None:
    body = register(client)
    with Session(get_engine()) as session:
        user = session.get(User, body["user"]["id"])
        assert user is not None
        user.disabled = True
        session.add(user)
        session.commit()

    assert client.get(f"{API}/auth/me", headers=auth(body["token"])).status_code == 401
    assert client.post(f"{API}/auth/login", json={"email": "ada@example.com", "password": GOOD_PASSWORD}).status_code == 401


# ======================================================================================
# What a request without a session may do
# ======================================================================================


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", f"{API}/properties"),
        ("post", f"{API}/properties"),
        ("get", f"{API}/properties/anything"),
        ("get", f"{API}/jobs/anything"),
        ("get", f"{API}/ai/restyle/styles"),
    ],
)
def test_the_private_api_refuses_an_anonymous_caller(client: TestClient, method: str, path: str) -> None:
    response = client.post(path, json={}) if method == "post" else client.get(path)
    assert response.status_code == 401, f"{method} {path} answered {response.status_code}"


@pytest.mark.parametrize("path", [f"{API}/health", f"{API}/auth/google/status"])
def test_the_public_api_does_not(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 200


def test_a_made_up_token_is_not_a_session(client: TestClient) -> None:
    assert client.get(f"{API}/auth/me", headers=auth("not-a-real-token")).status_code == 401


# ======================================================================================
# Ownership
# ======================================================================================


def test_one_account_cannot_see_or_touch_another_account_s_property(client: TestClient) -> None:
    ada = register(client, email="ada@example.com", name="Ada")
    grace = register(client, email="grace@example.com", name="Grace")

    created = client.post(f"{API}/properties", json={"title": "Ada'nın evi"}, headers=auth(ada["token"]))
    assert created.status_code == 201
    property_id = created.json()["id"]

    # Grace's list does not contain it...
    listed = client.get(f"{API}/properties", headers=auth(grace["token"]))
    assert listed.status_code == 200
    assert listed.json()["total"] == 0
    assert listed.json()["items"] == []

    # ...and asking for it by id says "not found" rather than "not yours", which would
    # confirm the id is real.
    for response in (
        client.get(f"{API}/properties/{property_id}", headers=auth(grace["token"])),
        client.patch(f"{API}/properties/{property_id}", json={"title": "mine now"}, headers=auth(grace["token"])),
        client.delete(f"{API}/properties/{property_id}", headers=auth(grace["token"])),
    ):
        assert response.status_code == 404
        assert response.json()["code"] == "property_not_found"

    # And it is still Ada's.
    assert client.get(f"{API}/properties/{property_id}", headers=auth(ada["token"])).status_code == 200


def test_a_room_is_reachable_only_through_its_own_owner(client: TestClient) -> None:
    ada = register(client, email="ada@example.com")
    grace = register(client, email="grace@example.com")
    property_id = client.post(f"{API}/properties", json={"title": "Ev"}, headers=auth(ada["token"])).json()["id"]
    scene_id = client.post(
        f"{API}/properties/{property_id}/rooms", json={"name": "Salon"}, headers=auth(ada["token"])
    ).json()["id"]

    assert client.get(f"{API}/scenes/{scene_id}", headers=auth(grace["token"])).status_code == 404
    assert client.get(f"{API}/scenes/{scene_id}", headers=auth(ada["token"])).status_code == 200


# ======================================================================================
# The upgrade path
# ======================================================================================


def test_the_first_account_adopts_the_properties_that_predate_accounts(client: TestClient) -> None:
    """A library that existed before there were logins must not become unreachable."""
    with Session(get_engine()) as session:
        session.add(Property(title="Eski ev", owner_id=None))
        session.commit()

    ada = register(client, email="ada@example.com")
    listed = client.get(f"{API}/properties", headers=auth(ada["token"]))
    assert [item["title"] for item in listed.json()["items"]] == ["Eski ev"]


def test_the_second_account_adopts_nothing(client: TestClient) -> None:
    with Session(get_engine()) as session:
        session.add(Property(title="Eski ev", owner_id=None))
        session.commit()
    register(client, email="ada@example.com")

    with Session(get_engine()) as session:
        session.add(Property(title="Sahipsiz kalan", owner_id=None))
        session.commit()

    grace = register(client, email="grace@example.com")
    assert client.get(f"{API}/properties", headers=auth(grace["token"])).json()["total"] == 0


# ======================================================================================
# Publishing
# ======================================================================================


def test_the_embed_link_works_only_while_the_property_is_published(client: TestClient) -> None:
    """`published` used to be a label nothing enforced; taking a property off air now takes it off air."""
    ada = register(client)
    created = client.post(f"{API}/properties", json={"title": "Ev"}, headers=auth(ada["token"])).json()
    token = created["public_token"]

    assert client.get(f"{API}/public/properties/{token}").status_code == 404

    client.patch(f"{API}/properties/{created['id']}", json={"published": True}, headers=auth(ada["token"]))
    assert client.get(f"{API}/public/properties/{token}").status_code == 200

    client.patch(f"{API}/properties/{created['id']}", json={"published": False}, headers=auth(ada["token"]))
    assert client.get(f"{API}/public/properties/{token}").status_code == 404


# ======================================================================================
# The primitives
# ======================================================================================


def test_a_stored_hash_never_contains_the_password_and_verifies_only_itself() -> None:
    stored = auth_lib.hash_password("correct-horse-battery")
    assert "correct-horse-battery" not in stored
    assert stored.startswith("scrypt$")
    assert auth_lib.verify_password("correct-horse-battery", stored)
    assert not auth_lib.verify_password("correct-horse-batterz", stored)


def test_two_hashes_of_one_password_differ(client: TestClient) -> None:
    """Salted. Identical hashes would tell an attacker which accounts share a password."""
    assert auth_lib.hash_password("correct-horse-battery") != auth_lib.hash_password("correct-horse-battery")


def test_a_password_less_account_cannot_be_signed_into_with_any_password() -> None:
    assert not auth_lib.verify_password("", None)
    assert not auth_lib.verify_password("anything at all", None)


def test_the_session_token_is_not_what_is_stored(client: TestClient) -> None:
    token = register(client)["token"]
    with Session(get_engine()) as session:
        row = session.exec(select(AuthSession)).one()
    assert row.token_hash != token
    assert row.token_hash == auth_lib.token_digest(token)


def test_google_sign_in_is_off_until_it_is_configured(client: TestClient) -> None:
    assert client.get(f"{API}/auth/google/status").json() == {"enabled": False}
    assert client.get(f"{API}/auth/google/start", follow_redirects=False).status_code == 501
