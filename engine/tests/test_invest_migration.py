"""The investment assistant's schema change, applied to a database that predates it.

Two additions and nothing else: a nullable JSON column on ``property`` (the agent's figures) and
a new ``advisor_usage`` table (the bill). Both ride the generic upgrade in ``db._migrate_sqlite``
— ``ADD COLUMN`` for the first, ``create_all`` for the second — so this pins what that has to
produce on a library that already holds listings: every row kept, the new column NULL (the
assistant off) for all of them, and the usage table's foreign key behaving like the model says.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlmodel import Session, SQLModel

from biosphere.core.config import get_settings
from biosphere.core.db import get_engine, init_db, reset_engine
from biosphere.core.models import AdvisorUsage, Property


@pytest.fixture
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "before.db"
    monkeypatch.setenv("BIOSPHERE_MEDIA_ROOT", str(tmp_path / "media"))
    monkeypatch.setenv("BIOSPHERE_DATABASE_URL", f"sqlite:///{path.as_posix()}")
    get_settings.cache_clear()
    reset_engine()
    yield path
    reset_engine()
    get_settings.cache_clear()


def _database_before_the_feature(path: Path) -> None:
    """Today's schema minus the assistant, holding one listing."""
    engine = get_engine()
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Property(id="p1", title="Old flat", public_token="tok-1"))
        session.commit()
    reset_engine()
    con = sqlite3.connect(path)
    con.execute("DROP TABLE advisor_usage")
    con.execute('ALTER TABLE "property" DROP COLUMN "investment"')
    con.commit()
    con.close()


def _columns(path: Path, table: str) -> dict[str, tuple[str, int]]:
    con = sqlite3.connect(path)
    rows = {r[1]: (r[2], r[3]) for r in con.execute(f'PRAGMA table_info("{table}")')}
    con.close()
    return rows


def test_the_upgrade_adds_the_column_and_the_table_and_keeps_every_row(db_path: Path) -> None:
    _database_before_the_feature(db_path)
    assert "investment" not in _columns(db_path, "property")

    init_db()

    columns = _columns(db_path, "property")
    assert columns["investment"] == ("JSON", 0), "nullable JSON, so existing rows simply have it off"
    usage = _columns(db_path, "advisor_usage")
    assert {"property_id", "conversation_id", "cost_usd", "web_search_requests", "created_at"} <= set(usage)

    con = sqlite3.connect(db_path)
    assert con.execute("SELECT id, investment FROM property").fetchall() == [("p1", None)]
    indexes = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='advisor_usage'")}
    con.close()
    assert {"ix_advisor_usage_property_id", "ix_advisor_usage_created_at", "ix_advisor_usage_conversation_id"} <= indexes


def test_a_deleted_listing_keeps_its_bill(db_path: Path) -> None:
    """``SET NULL``, not ``CASCADE``: the money was spent, and the installation's daily cap
    must keep counting it after the listing is gone."""
    _database_before_the_feature(db_path)
    init_db()
    with Session(get_engine()) as session:
        session.add(AdvisorUsage(property_id="p1", conversation_id="c1", model="claude-opus-5-5", cost_usd=0.25))
        session.commit()
        session.delete(session.get(Property, "p1"))
        session.commit()
    con = sqlite3.connect(db_path)
    assert con.execute("SELECT property_id, cost_usd FROM advisor_usage").fetchall() == [(None, 0.25)]
    con.close()


def test_the_upgrade_is_idempotent(db_path: Path) -> None:
    _database_before_the_feature(db_path)
    init_db()
    reset_engine()
    init_db()  # second boot: must not raise or duplicate anything
    assert _columns(db_path, "property")["investment"] == ("JSON", 0)
