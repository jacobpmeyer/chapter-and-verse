"""Database backups: dated pg_dump archives outside the project.

`pg_dump` runs inside the Postgres container (via `docker compose exec`), so its
version always matches the server's. A pg_dump older than the server refuses to
run, which is the case with a typical Homebrew/mise install.

Each backup is written to a temporary name, checked with `pg_restore --list`,
and only then renamed into place, so a failed or partial dump never looks like
a good backup.
"""

from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from config import settings

PROJECT_DIR = Path(__file__).parent
PREFIX = "chapter-and-verse-"
NAME_RE = re.compile(rf"^{PREFIX}\d{{8}}-\d{{6}}(-[\w-]+)?\.dump$")
EXPECTED_TABLES = ("books", "chapters", "chunks", "api_calls")


class BackupError(RuntimeError):
    pass


def backup_dir() -> Path:
    return Path(os.getenv("BACKUP_DIR", "~/Backups/chapter-and-verse")).expanduser()


def backup_name(now: datetime, label: str | None = None) -> str:
    suffix = f"-{re.sub(r'[^A-Za-z0-9_-]+', '-', label).strip('-')}" if label else ""
    return f"{PREFIX}{now:%Y%m%d-%H%M%S}{suffix}.dump"


def list_backups(directory: Path) -> list[Path]:
    """This project's backups in `directory`, oldest first (names sort by time)."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if NAME_RE.match(p.name))


def prune(directory: Path, keep: int) -> list[Path]:
    """Delete all but the newest `keep` backups; returns what was deleted."""
    if keep < 1:
        raise ValueError("keep must be at least 1")
    old = list_backups(directory)[:-keep]
    for p in old:
        p.unlink()
    return old


def _db_identity() -> tuple[str, str]:
    url = urlparse(settings.database_url)
    user, dbname = url.username, url.path.lstrip("/")
    if not user or not dbname:
        raise BackupError("DATABASE_URL must include a user and database name")
    return user, dbname


def _compose(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", "compose", "exec", "-T", "db", *args], cwd=PROJECT_DIR, **kwargs)


def create_backup(label: str | None = None, directory: Path | None = None) -> Path:
    directory = directory or backup_dir()
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / backup_name(datetime.now(), label)
    partial = final.with_suffix(".dump.partial")
    user, dbname = _db_identity()
    try:
        with partial.open("wb") as out:
            dump = _compose("pg_dump", "-U", user, "-d", dbname, "--format=custom", stdout=out,
                            stderr=subprocess.PIPE)
        if dump.returncode != 0:
            raise BackupError(f"pg_dump failed: {dump.stderr.decode().strip() or 'is the database running?'}")
        with partial.open("rb") as archive:
            listing = _compose("pg_restore", "--list", stdin=archive, capture_output=True)
        contents = listing.stdout.decode()
        missing = [t for t in EXPECTED_TABLES if f"TABLE DATA public {t} " not in contents]
        if listing.returncode != 0 or missing:
            raise BackupError(f"backup failed verification (missing table data: {', '.join(missing) or 'n/a'})")
        partial.rename(final)
        return final
    except FileNotFoundError as e:
        raise BackupError("docker isn't installed or isn't on PATH") from e
    finally:
        partial.unlink(missing_ok=True)


def restore_command(path: Path) -> str:
    user, dbname = _db_identity()
    return (f'docker compose exec -T db pg_restore -U {user} -d {dbname} --clean --if-exists '
            f'--single-transaction < "{path}"')
