"""
Database initialisation, session management, and schema versioning.

Usage:
    from core.memory.database import get_session, init_db

    init_db()   # call once at startup
    with get_session() as session:
        session.add(alarm)
        session.commit()
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from loguru import logger
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from core.memory.models import Base

# ── Database file ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent.parent
DB_PATH = ROOT / "trading_bot.db"
DB_URL = f"sqlite:///{DB_PATH}"

# ── Engine ────────────────────────────────────────────────────────────────────

engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False},   # needed for async access
    echo=False,
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):
    """Enable WAL mode and foreign keys for every new SQLite connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


# ── Public API ────────────────────────────────────────────────────────────────


def init_db() -> None:
    """Create all tables if they don't exist. Safe to call on every startup."""
    Base.metadata.create_all(bind=engine)
    logger.info(f"Database initialised at {DB_PATH}")


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Provide a transactional database session."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_path() -> Path:
    return DB_PATH


def table_exists(table_name: str) -> bool:
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name=:name"
            ),
            {"name": table_name},
        )
        return result.fetchone() is not None
