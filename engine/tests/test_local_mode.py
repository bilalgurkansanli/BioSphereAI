"""Local mode: the open-source default, where your own dashboard needs no sign-in."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from biosphere.core.config import get_settings

from .test_api import API, settings  # noqa: F401 - fixture


@pytest.fixture()
def local_client(settings, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:  # noqa: F811
    monkeypatch.setenv("BIOSPHERE_AUTH_MODE", "local")
    get_settings.cache_clear()
    from biosphere.api.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
    get_settings.cache_clear()


def test_the_dashboard_opens_without_signing_in(local_client: TestClient) -> None:
    me = local_client.get(f"{API}/auth/me")
    assert me.status_code == 200
    assert me.json()["local"] is True

    created = local_client.post(f"{API}/properties", json={"title": "Kadıköy 3+1"})
    assert created.status_code == 201
    listed = local_client.get(f"{API}/properties")
    assert [p["id"] for p in listed.json()["items"]] == [created.json()["id"]]


def test_local_mode_acts_as_the_oldest_account(local_client: TestClient) -> None:
    registered = local_client.post(
        f"{API}/auth/register",
        json={"email": "ada@example.com", "name": "Ada", "password": "correct-horse-battery"},
    )
    assert registered.status_code == 201
    local_client.cookies.clear()
    me = local_client.get(f"{API}/auth/me")
    assert me.json()["email"] == "ada@example.com" and me.json()["local"] is True
