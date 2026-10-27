"""The SQLite job-table rebuild (``db._migrate_sqlite``) against real pre-walkthrough databases.

Two facts about SQLite shaped the rebuild and both are pinned here: a renamed table keeps
its index NAMES (so creating the new table's indexes collides until the old ones are
dropped), and DDL commits implicitly (so a rebuild that died halfway leaves ``job__old``
next to an empty ``job`` and must resume rather than fail on the next start).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.schema import CreateTable
from sqlmodel import Session, SQLModel

from biosphere.core.config import get_settings
from biosphere.core.db import get_engine, init_db, reset_engine
from biosphere.core.models import Property, Scene, User


@pytest.fixture
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "old.db"
    monkeypatch.setenv("BIOSPHERE_MEDIA_ROOT", str(tmp_path / "media"))
    monkeypatch.setenv("BIOSPHERE_DATABASE_URL", f"sqlite:///{path.as_posix()}")
    get_settings.cache_clear()
    reset_engine()
    yield path
    reset_engine()
    get_settings.cache_clear()


def _old_job_table(path: Path) -> None:
    """A ``job`` table as the pre-walkthrough engine created it: ``scene_id NOT NULL`` and
    the three named indexes, holding one finished job."""
    engine = get_engine()
    # The rest of the schema, with the property and scene the job points at (foreign keys are on).
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Property(id="t1", title="Old flat"))
        session.commit()
        session.add(Scene(id="s1", property_id="t1", name="Salon"))
        session.commit()
    ddl = str(CreateTable(SQLModel.metadata.tables["job"]).compile(engine))
    assert "scene_id VARCHAR(32)" in ddl
    ddl = ddl.replace("scene_id VARCHAR(32)", "scene_id VARCHAR(32) NOT NULL", 1)
    con = sqlite3.connect(path)
    con.executescript(
        "DROP TABLE job;\n"
        + ddl
        + ";\nCREATE INDEX ix_job_property_id ON job (property_id);"
        + "\nCREATE INDEX ix_job_status ON job (status);"
        + "\nCREATE INDEX ix_job_scene_id ON job (scene_id);"
    )
    con.execute(
        "INSERT INTO job (id, property_id, scene_id, kind, status, stage, progress, stage_progress, message, message_tr, "
        "eta_seconds, metrics, error, quality_mode, capture_kind, created_at) VALUES "
        "('j1', 't1', 's1', 'stitch', 'done', 'done', 1.0, 1.0, '', '', NULL, '{}', NULL, 'balanced', 'video', '2026-01-01 00:00:00')"
    )
    con.commit()
    con.close()


def _describe(path: Path) -> tuple[list[str], list[str], int, int]:
    con = sqlite3.connect(path)
    tables = sorted(r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'job%'"))
    indexes = sorted(
        r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='job' AND name LIKE 'ix_%'")
    )
    notnull = next(int(r[3]) for r in con.execute("PRAGMA table_info(job)") if r[1] == "scene_id")
    rows = int(con.execute("SELECT COUNT(*) FROM job").fetchone()[0])
    con.close()
    return tables, indexes, notnull, rows


def test_rebuild_drops_the_renamed_tables_indexes_first(db_path: Path) -> None:
    _old_job_table(db_path)
    init_db()
    tables, indexes, notnull, rows = _describe(db_path)
    assert tables == ["job"]
    assert indexes == ["ix_job_property_id", "ix_job_scene_id", "ix_job_status"]
    assert notnull == 0
    assert rows == 1


def test_rebuild_resumes_after_an_interrupted_attempt(db_path: Path) -> None:
    _old_job_table(db_path)
    # What a crash between "rename" and "create index" leaves behind (DDL auto-commits).
    con = sqlite3.connect(db_path)
    con.execute("ALTER TABLE job RENAME TO job__old")
    con.execute("CREATE TABLE job (id VARCHAR(32) PRIMARY KEY, property_id VARCHAR(32), scene_id VARCHAR(32))")
    con.commit()
    con.close()
    init_db()
    tables, indexes, notnull, rows = _describe(db_path)
    assert tables == ["job"]
    assert indexes == ["ix_job_property_id", "ix_job_scene_id", "ix_job_status"]
    assert notnull == 0
    assert rows == 1


def test_rebuild_refuses_to_guess_when_both_tables_hold_rows(db_path: Path) -> None:
    _old_job_table(db_path)
    con = sqlite3.connect(db_path)
    con.execute("ALTER TABLE job RENAME TO job__old")
    con.execute("CREATE TABLE job (id VARCHAR(32) PRIMARY KEY, property_id VARCHAR(32), scene_id VARCHAR(32))")
    con.execute("INSERT INTO job (id, property_id, scene_id) VALUES ('j2', 't1', 's1')")
    con.commit()
    con.close()
    with pytest.raises(RuntimeError, match="merge them by hand"):
        init_db()


# --------------------------------------------------------------------------------------
# tour -> property
# --------------------------------------------------------------------------------------


def _v2_database(path: Path) -> None:
    """A database as v2 left it: a ``tour`` table, ``scene.tour_id``, and a panorama whose
    URLs point into ``/media/tours/``. Only the columns the rename touches are declared —
    ``init_db`` adds the rest through the ADD COLUMN pass, which is the real upgrade path."""
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE tour (
            id VARCHAR(32) NOT NULL PRIMARY KEY,
            title VARCHAR(200) NOT NULL,
            public_token VARCHAR(64) NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL
        );
        CREATE UNIQUE INDEX ix_tour_public_token ON tour (public_token);
        CREATE TABLE scene (
            id VARCHAR(32) NOT NULL PRIMARY KEY,
            tour_id VARCHAR(32) NOT NULL REFERENCES tour(id) ON DELETE CASCADE,
            name VARCHAR(200) NOT NULL,
            panorama JSON
        );
        CREATE INDEX ix_scene_tour_id ON scene (tour_id);
        """
    )
    con.execute(
        "INSERT INTO tour (id, title, public_token, created_at, updated_at) "
        "VALUES ('t1', 'Bahceli Villa', 'tok1', '2026-01-01 00:00:00', '2026-01-01 00:00:00')"
    )
    con.execute(
        "INSERT INTO scene (id, tour_id, name, panorama) VALUES ('s1', 't1', 'Salon', ?)",
        ('{"levels": [{"url": "/media/tours/t1/scenes/s1/pano_hd.jpg"}], '
         '"preview_url": "/media/tours/t1/scenes/s1/pano_preview.webp"}',),
    )
    con.commit()
    con.close()


def test_the_tour_table_its_columns_and_its_media_urls_are_renamed_together(db_path: Path) -> None:
    _v2_database(db_path)
    media = get_settings().media_root
    old_tree = media / "tours" / "t1" / "scenes" / "s1"
    old_tree.mkdir(parents=True, exist_ok=True)
    (old_tree / "pano_hd.jpg").write_bytes(b"jpeg")

    init_db()

    con = sqlite3.connect(db_path)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "property" in tables and "tour" not in tables
    columns = {r[1] for r in con.execute('PRAGMA table_info("scene")')}
    assert "property_id" in columns and "tour_id" not in columns
    # The row survived the rename with its foreign key intact...
    assert con.execute("SELECT property_id FROM scene WHERE id = 's1'").fetchone()[0] == "t1"
    # ...and every URL inside the JSON document moved with the directory.
    panorama = con.execute("SELECT panorama FROM scene WHERE id = 's1'").fetchone()[0]
    assert "/media/tours/" not in panorama
    assert panorama.count("/media/properties/t1/scenes/s1/") == 2
    # The index named after the old column is gone, replaced by the one the model declares.
    indexes = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='scene'")}
    con.close()
    assert "ix_scene_tour_id" not in indexes
    assert "ix_scene_property_id" in indexes
    tokens = {r[0] for r in sqlite3.connect(db_path).execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='property'")}
    assert "ix_property_public_token" in tokens and "ix_tour_public_token" not in tokens

    assert (media / "properties" / "t1" / "scenes" / "s1" / "pano_hd.jpg").read_bytes() == b"jpeg"
    assert not (media / "tours").exists()


def test_the_rename_is_a_no_op_on_a_database_that_already_speaks_property(db_path: Path) -> None:
    _v2_database(db_path)
    init_db()
    reset_engine()
    init_db()  # second boot: must not raise, must not undo anything
    con = sqlite3.connect(db_path)
    assert con.execute("SELECT COUNT(*) FROM property").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM scene").fetchone()[0] == 1
    con.close()


def test_a_column_added_by_the_migration_keeps_its_foreign_key(db_path: Path) -> None:
    """An upgraded database must behave like a fresh one, not merely look like it.

    ``property.owner_id`` reaches an existing installation through the ADD COLUMN pass rather
    than through ``create_all``. When that pass dropped the REFERENCES clause, the column
    existed, every query worked, and the only symptom was that deleting an account left its
    properties behind — on a fresh database the same delete cascaded, so nothing in
    development ever showed it.
    """
    _v2_database(db_path)
    init_db()

    with Session(get_engine()) as session:
        session.add(User(id="u1", email="ada@example.com", name="Ada", password_hash="x"))
        session.commit()
        prop = session.get(Property, "t1")
        assert prop is not None
        prop.owner_id = "u1"
        session.add(prop)
        session.commit()

    con = sqlite3.connect(db_path)
    keys = {(r[3], r[2], r[6]) for r in con.execute('PRAGMA foreign_key_list("property")')}
    con.close()
    assert ("owner_id", "user", "CASCADE") in keys

    # And the constraint is live, not merely declared.
    with Session(get_engine()) as session:
        session.delete(session.get(User, "u1"))
        session.commit()
    with Session(get_engine()) as session:
        assert session.get(Property, "t1") is None
