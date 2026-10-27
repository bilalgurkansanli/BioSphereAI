"""The dashboard's "own API key" setting: saved locally, never echoed, owner-only."""

from __future__ import annotations

from fastapi.testclient import TestClient

from biosphere.core.config import get_settings

from .test_api import API, client, settings  # noqa: F401 - fixtures

KEY = "sk-ant-api03-" + "x" * 40 + "a1b2"


def _register(client: TestClient, email: str) -> dict[str, str]:  # noqa: F811
    r = client.post(
        f"{API}/auth/register",
        json={"email": email, "name": email.split("@")[0], "password": "correct-horse-battery"},
    )
    assert r.status_code == 201, r.text
    client.cookies.clear()
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_a_saved_key_is_used_and_never_returned(client: TestClient, monkeypatch) -> None:  # noqa: F811
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    owner = _register(client, "owner@example.com")
    before = client.get(f"{API}/settings/ai", headers=owner).json()
    assert before["configured"] is False and before["source"] == "none"

    saved = client.put(f"{API}/settings/ai", json={"api_key": KEY}, headers=owner)
    assert saved.status_code == 200
    body = saved.json()
    assert body == {**body, "configured": True, "source": "dashboard", "hint": "sk-ant-…a1b2"}
    assert KEY not in saved.text
    assert get_settings().listing_key == KEY and get_settings().advisor_key == KEY

    cleared = client.delete(f"{API}/settings/ai", headers=owner).json()
    assert cleared["configured"] is False and get_settings().listing_key == ""


def test_a_malformed_key_is_refused(client: TestClient) -> None:  # noqa: F811
    owner = _register(client, "owner@example.com")
    r = client.put(f"{API}/settings/ai", json={"api_key": "hello@example.com"}, headers=owner)
    assert r.status_code == 422


def test_only_the_owner_changes_the_key_in_accounts_mode(client: TestClient) -> None:  # noqa: F811
    _register(client, "owner@example.com")
    other = _register(client, "other@example.com")
    assert client.get(f"{API}/settings/ai", headers=other).json()["can_edit"] is False
    r = client.put(f"{API}/settings/ai", json={"api_key": KEY}, headers=other)
    assert r.status_code == 403


def test_settings_need_an_account(client: TestClient) -> None:  # noqa: F811
    client.cookies.clear()
    assert client.get(f"{API}/settings/ai").status_code == 401
