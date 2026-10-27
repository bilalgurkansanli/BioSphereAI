"""Capture links: a QR code that can film one property and do nothing else.

The client is ANONYMOUS, as in ``test_auth``: every request names its credential.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from biosphere.core.db import get_engine
from biosphere.core.models import CaptureGrant, Job, Property

from .test_api import API, client, settings  # noqa: F401 - fixtures
from .test_auth import auth, register


def _property(test_client: TestClient, token: str, title: str = "Kadıköy 3+1") -> str:
    created = test_client.post(f"{API}/properties", json={"title": title}, headers=auth(token))
    assert created.status_code == 201, created.text
    return created.json()["id"]


def _link(test_client: TestClient, token: str, property_id: str) -> str:
    minted = test_client.post(f"{API}/properties/{property_id}/capture-link", headers=auth(token))
    assert minted.status_code == 201, minted.text
    body = minted.json()
    assert body["property_id"] == property_id
    expires = datetime.fromisoformat(body["expires_at"].replace("Z", "+00:00"))
    assert timedelta(hours=11) < expires - datetime.now(UTC) <= timedelta(hours=12, minutes=1)
    return body["token"]


@pytest.fixture()
def owner(client: TestClient) -> dict:  # noqa: F811 - pytest fixture injection
    session = register(client)
    token = session["token"]
    home = _property(client, token)
    other = _property(client, token, "Çankaya 2+1")
    return {"token": token, "home": home, "other": other, "grant": _link(client, token, home)}


def test_a_capture_link_reads_its_own_property(client: TestClient, owner: dict) -> None:  # noqa: F811
    got = client.get(f"{API}/properties/{owner['home']}", headers=auth(owner["grant"]))
    assert got.status_code == 200
    assert got.json()["id"] == owner["home"]


def test_a_capture_link_can_create_a_room_in_its_property_only(client: TestClient, owner: dict) -> None:  # noqa: F811
    room = client.post(f"{API}/properties/{owner['home']}/rooms", json={"name": "Salon"}, headers=auth(owner["grant"]))
    assert room.status_code == 201, room.text
    elsewhere = client.post(
        f"{API}/properties/{owner['other']}/rooms", json={"name": "Salon"}, headers=auth(owner["grant"])
    )
    assert elsewhere.status_code == 401


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/properties"),
        ("GET", "/auth/me"),
        ("PATCH", "/properties/{home}"),
        ("DELETE", "/properties/{home}"),
        ("GET", "/properties/{other}"),
        ("POST", "/properties/{home}/capture-link"),
        ("DELETE", "/properties/{home}/capture-links"),
        ("GET", "/properties/{home}/plan"),
        ("GET", "/properties/{home}/listing"),
        ("DELETE", "/properties/{home}/walkthrough"),
        ("GET", "/ai/restyle/styles"),
    ],
)
def test_a_capture_link_is_refused_everywhere_else(
    client: TestClient,  # noqa: F811
    owner: dict,
    method: str,
    path: str,
) -> None:
    url = API + path.format(home=owner["home"], other=owner["other"])
    response = client.request(method, url, headers=auth(owner["grant"]), json={"title": "x"} if method == "PATCH" else None)
    assert response.status_code == 401, (method, path, response.status_code, response.text[:200])
    # And nothing was changed by the attempt.
    still = client.get(f"{API}/properties/{owner['home']}", headers=auth(owner["token"]))
    assert still.status_code == 200 and still.json()["title"] == "Kadıköy 3+1"


def test_scene_and_job_ids_must_belong_to_the_grant_s_property(client: TestClient, owner: dict) -> None:  # noqa: F811
    mine = client.post(f"{API}/properties/{owner['home']}/rooms", json={"name": "Mutfak"}, headers=auth(owner["token"]))
    theirs = client.post(f"{API}/properties/{owner['other']}/rooms", json={"name": "Mutfak"}, headers=auth(owner["token"]))
    assert mine.status_code == theirs.status_code == 201
    assert client.get(f"{API}/scenes/{mine.json()['id']}", headers=auth(owner["grant"])).status_code == 200
    # The owner's OTHER property is the same account — exactly what the scope check is for.
    assert client.get(f"{API}/scenes/{theirs.json()['id']}", headers=auth(owner["grant"])).status_code == 401

    with Session(get_engine()) as session:
        job = Job(property_id=owner["other"], kind="stitch", status="done")
        session.add(job)
        session.commit()
        job_id = job.id
    assert client.get(f"{API}/jobs/{job_id}", headers=auth(owner["grant"])).status_code == 401


def test_an_expired_link_is_refused_and_deleted(client: TestClient, owner: dict) -> None:  # noqa: F811
    with Session(get_engine()) as session:
        grant = session.exec(select(CaptureGrant).where(CaptureGrant.property_id == owner["home"])).one()
        grant.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        session.add(grant)
        session.commit()
    assert client.get(f"{API}/properties/{owner['home']}", headers=auth(owner["grant"])).status_code == 401
    with Session(get_engine()) as session:
        assert session.exec(select(CaptureGrant).where(CaptureGrant.property_id == owner["home"])).first() is None


def test_revoking_kills_every_link_of_the_property(client: TestClient, owner: dict) -> None:  # noqa: F811
    second = _link(client, owner["token"], owner["home"])
    revoked = client.delete(f"{API}/properties/{owner['home']}/capture-links", headers=auth(owner["token"]))
    assert revoked.status_code == 204
    for token in (owner["grant"], second):
        assert client.get(f"{API}/properties/{owner['home']}", headers=auth(token)).status_code == 401


def test_a_link_dies_when_its_creator_no_longer_owns_the_property(client: TestClient, owner: dict) -> None:  # noqa: F811
    colleague = register(client, email="ayse@example.com", name="Ayşe")
    with Session(get_engine()) as session:
        prop = session.get(Property, owner["home"])
        assert prop is not None
        prop.owner_id = colleague["user"]["id"]
        session.add(prop)
        session.commit()
    assert client.get(f"{API}/properties/{owner['home']}", headers=auth(owner["grant"])).status_code == 401


def test_only_the_owner_can_mint_a_link(client: TestClient, owner: dict) -> None:  # noqa: F811
    stranger = register(client, email="mehmet@example.com", name="Mehmet")
    minted = client.post(f"{API}/properties/{owner['home']}/capture-link", headers=auth(stranger["token"]))
    assert minted.status_code == 404
    # `register` signed the client in with a cookie too; an anonymous caller has neither.
    client.cookies.clear()
    anonymous = client.post(f"{API}/properties/{owner['home']}/capture-link")
    assert anonymous.status_code == 401


def test_the_token_is_not_what_is_stored(client: TestClient, owner: dict) -> None:  # noqa: F811
    with Session(get_engine()) as session:
        rows = session.exec(select(CaptureGrant)).all()
    assert rows and all(row.token_hash != owner["grant"] and len(row.token_hash) == 64 for row in rows)
