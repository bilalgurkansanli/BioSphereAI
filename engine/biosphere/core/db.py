"""SQLite engine, session plumbing and schema creation.

Two SQLite details are load-bearing here and both are easy to get wrong:

1. ``PRAGMA foreign_keys=ON`` is **per connection** and defaults to *off*. Without it the
   ``ON DELETE CASCADE`` declared on the scene/hotspot/job foreign keys is silently ignored
   and deleting a property leaves orphan rows behind.
2. ``PRAGMA journal_mode=WAL`` lets the SSE readers and the stitch worker touch the database
   at the same time instead of serialising behind a global write lock. Combined with
   ``check_same_thread=False`` (the worker thread owns its own session) this is what makes
   the single-process design safe.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import JSON
from sqlmodel import Session, SQLModel, create_engine

from biosphere.core import models as _models  # noqa: F401  (table registration side effect)
from biosphere.core.config import get_settings

__all__ = [
    "get_engine",
    "reset_engine",
    "init_db",
    "session_scope",
    "get_session",
]

_engine: Engine | None = None


def _install_pragmas(engine: Engine) -> None:
    """Apply the per-connection PRAGMAs every time the pool opens a connection."""

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection: Any, _record: Any) -> None:  # pragma: no cover - driver hook
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=10000")
        finally:
            cursor.close()


def get_engine() -> Engine:
    """Process-wide engine, created on first use so tests can repoint the media root first."""
    global _engine
    if _engine is not None:
        return _engine

    settings = get_settings()
    url = settings.database_url
    connect_args: dict[str, Any] = {}
    kwargs: dict[str, Any] = {}

    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        # The file lives under media_root/db; make sure the directory exists even if the
        # settings validator ran against a different root earlier in the process.
        if url.startswith("sqlite:///") and ":memory:" not in url:
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        else:
            # An in-memory database must share one connection or every session sees a fresh,
            # empty schema.
            kwargs["poolclass"] = StaticPool

    _engine = create_engine(url, echo=False, connect_args=connect_args, **kwargs)
    if url.startswith("sqlite"):
        _install_pragmas(_engine)
    return _engine


def reset_engine() -> None:
    """Dispose the cached engine. Used by tests that repoint the database between cases."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


def init_db() -> None:
    """Create any missing tables and columns. Safe to call on every boot.

    ``create_all`` never alters an existing table, and this project ships no migration
    tool, so the two schema changes the walkthrough feature made are applied here by hand:
    columns added to ``scene`` and ``job`` (``ALTER TABLE ... ADD COLUMN``, which SQLite
    supports) and ``job.scene_id`` becoming nullable, which SQLite does not support in place
    and is done by rebuilding the table. Job rows are progress history, not content, so the
    rebuild costs nothing a user would notice.
    """
    engine = get_engine()
    if engine.url.get_backend_name() == "sqlite":
        # Strictly before ``create_all``: that call would otherwise create an empty
        # ``property`` table beside the populated ``tour`` one and the rename below would
        # then refuse, leaving a database with both and content in neither.
        _rename_tour_to_property(engine)
    SQLModel.metadata.create_all(engine)
    if engine.url.get_backend_name() == "sqlite":
        _migrate_sqlite(engine)


def _rename_tour_to_property(engine: Engine) -> None:
    """v2 called the listing entity a *tour*; it is a *property* ("ev"). One-way, idempotent.

    THE THREE STEPS BELOW ARE ONE FACT, not three migrations. The media directory, the paths
    on disk and the ``/media/...`` URLs written into the ``panorama``/``walkthrough`` JSON
    columns all spell the same word, so renaming any one of them alone turns every panorama
    in the library into a 404. They are done together, inside one transaction for the parts
    SQLite can hold in one, and the directory is moved first: a move that fails leaves the
    database still pointing at a directory that still exists.
    """
    from sqlalchemy import text

    settings = get_settings()
    old_media = settings.media_root / "tours"
    new_media = settings.properties_dir
    if old_media.is_dir():
        # ``Settings._ensure_media_root`` has already created the new directory, empty, so
        # the plain rename would fail on Windows. An empty one is removed first; a new
        # directory that already holds something is a library that has been migrated (or a
        # collision this code must not resolve on its own), so the old tree is left alone.
        if not new_media.exists() or not any(new_media.iterdir()):
            if new_media.exists():
                new_media.rmdir()
            old_media.rename(new_media)

    with engine.begin() as conn:
        tables = {r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type = 'table'"))}
        if "tour" in tables and "property" not in tables:
            conn.execute(text('ALTER TABLE "tour" RENAME TO "property"'))
            # SQLite rewrites the foreign keys that referenced ``tour(id)`` for us, and
            # ``RENAME COLUMN`` rewrites the index definitions -- but it keeps their old
            # NAMES, so ``ix_scene_tour_id`` survives, now indexing ``property_id``.
            for table in ("scene", "job", "walkthrough"):
                if table not in tables:
                    continue
                columns = {r[1] for r in conn.execute(text(f'PRAGMA table_info("{table}")'))}
                if "tour_id" in columns and "property_id" not in columns:
                    conn.execute(text(f'ALTER TABLE "{table}" RENAME COLUMN "tour_id" TO "property_id"'))

            # The stale index NAMES go; the replacements are created at the end of
            # ``_migrate_sqlite``, once the ADD COLUMN pass has put every column the model
            # declares in place. Creating them here would try to index ``scene.status`` on a
            # v2 table that does not have it yet.
            stale = conn.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type = 'index' "
                    "AND (name LIKE 'ix\\_tour\\_%' ESCAPE '\\' OR name LIKE 'ix\\_%\\_tour\\_id' ESCAPE '\\')"
                )
            ).fetchall()
            for (index_name,) in stale:
                conn.execute(text(f'DROP INDEX IF EXISTS "{index_name}"'))

        # The URLs. Every text-ish column of every table, because a media URL can sit inside
        # a JSON document (``scene.panorama``, ``scene.walkthrough``) as easily as in a plain
        # column (``scene_photo.media_url``), and a targeted list would rot the next time one
        # is added.
        for table in conn.execute(
            text("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")
        ).fetchall():
            name = table[0]
            for row in conn.execute(text(f'PRAGMA table_info("{name}")')).fetchall():
                column, ddl_type = row[1], (row[2] or "").upper()
                if not any(kind in ddl_type for kind in ("CHAR", "TEXT", "JSON", "CLOB")):
                    continue
                conn.execute(
                    text(
                        f'UPDATE "{name}" SET "{column}" = REPLACE("{column}", :old, :new) '
                        f'WHERE "{column}" LIKE :like'
                    ),
                    {"old": "/media/tours/", "new": "/media/properties/", "like": "%/media/tours/%"},
                )


def _migrate_sqlite(engine: Engine) -> None:
    from sqlalchemy import text

    with engine.begin() as conn:
        # 1. Missing columns on existing tables.
        for table in SQLModel.metadata.sorted_tables:
            rows = conn.execute(text(f'PRAGMA table_info("{table.name}")')).fetchall()
            if not rows:
                continue
            present = {r[1] for r in rows}
            for column in table.columns:
                if column.name in present:
                    continue
                ddl_type = column.type.compile(dialect=engine.dialect)
                default = ""
                if not column.nullable:
                    if isinstance(column.type, JSON):
                        default = " DEFAULT '[]'"
                    elif column.default is not None and getattr(column.default, "arg", None) is not None and not callable(column.default.arg):
                        arg = column.default.arg
                        default = f" DEFAULT '{arg}'" if isinstance(arg, str) else f" DEFAULT {arg}"
                    else:
                        default = " DEFAULT ''"
                null = "" if column.nullable else " NOT NULL"
                # THE FOREIGN KEY HAS TO COME WITH THE COLUMN, or the schema an upgraded
                # installation ends up with is quietly not the schema `create_all` writes on
                # a fresh one. That difference is invisible until something depends on it:
                # ``property.owner_id`` was added by this loop without its REFERENCES clause,
                # so deleting an account left its properties behind as rows pointing at a user
                # that no longer existed — on a developer's fresh database the same delete
                # cascaded correctly, which is the worst possible way for a bug to behave.
                #
                # SQLite accepts REFERENCES in ADD COLUMN only for a column that defaults to
                # NULL, which is exactly the shape of every foreign key added after the fact.
                references = ""
                if column.foreign_keys and column.nullable and not default:
                    target = next(iter(column.foreign_keys))
                    on_delete = f" ON DELETE {target.ondelete}" if target.ondelete else ""
                    references = f' REFERENCES "{target.column.table.name}"("{target.column.name}"){on_delete}'
                conn.execute(
                    text(
                        f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" '
                        f"{ddl_type}{null}{default}{references}"
                    )
                )
        # 2. job.scene_id must be nullable for property-level jobs. SQLite cannot alter a column,
        # so the table is rebuilt: rename, create afresh, copy, drop. Two things SQLite does
        # here that a rebuild must expect: the renamed table KEEPS its index names, so the
        # new table's ``ix_job_*`` cannot be created until they are dropped; and DDL commits
        # implicitly, so an attempt that died halfway leaves ``job__old`` next to an empty
        # ``job`` — which is also the state to resume from, not an error.
        tables = {r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type = 'table'"))}
        rows = conn.execute(text('PRAGMA table_info("job")')).fetchall() if "job" in tables else []
        scene_col = next((r for r in rows if r[1] == "scene_id"), None)
        resume = "job__old" in tables
        if resume or (scene_col is not None and int(scene_col[3]) == 1):  # notnull flag
            if resume:
                if "job" in tables:
                    n_new = int(conn.execute(text('SELECT COUNT(*) FROM "job"')).scalar() or 0)
                    if n_new:
                        raise RuntimeError(
                            "job and job__old both hold rows: a previous migration of the job table "
                            "was interrupted after data was copied; merge them by hand"
                        )
                    conn.execute(text('DROP TABLE "job"'))
            else:
                conn.execute(text('ALTER TABLE "job" RENAME TO "job__old"'))
            stale = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'job__old' AND name NOT LIKE 'sqlite_%'")
            ).fetchall()
            for (index_name,) in stale:
                conn.execute(text(f'DROP INDEX IF EXISTS "{index_name}"'))
            SQLModel.metadata.tables["job"].create(conn)
            new_cols = [c.name for c in SQLModel.metadata.tables["job"].columns]
            old_cols = {r[1] for r in conn.execute(text('PRAGMA table_info("job__old")')).fetchall()}
            cols = ", ".join(f'"{c}"' for c in new_cols if c in old_cols)
            conn.execute(text(f'INSERT INTO "job" ({cols}) SELECT {cols} FROM "job__old"'))
            conn.execute(text('DROP TABLE "job__old"'))

        # 3. Indexes the model declares but the database does not have. ``create_all`` checks
        # for the TABLE and skips everything about one that already exists, indexes included,
        # so an index added to an existing table by a later version — or dropped above with
        # the tour name it was born with — would otherwise never appear. Last, because it
        # needs both the columns step 1 added and the table step 2 rebuilt.
        present = {
            r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type = 'index'"))
        }
        existing = {r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type = 'table'"))}
        for table in SQLModel.metadata.sorted_tables:
            if table.name not in existing:
                continue
            for index in table.indexes:
                if index.name not in present:
                    index.create(conn)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session for non-request code (the worker, the CLI).

    Commits on success, rolls back on any exception — the worker must never leave a job row
    half-updated when the pipeline throws.
    """
    session = Session(get_engine())
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency. Commit is explicit in the routes so handlers stay readable."""
    with Session(get_engine()) as session:
        yield session
