"""backup.py: file naming, listing, and pruning. These are the parts that
delete files, so they're tested without touching Docker or a real database."""

from datetime import datetime

import pytest

import backup as B

NOW = datetime(2026, 9, 30, 18, 24, 26)


def test_names_are_dated_and_sort_chronologically():
    assert B.backup_name(NOW) == "chapter-and-verse-20260930-182426.dump"
    assert B.backup_name(NOW, "before redo!") == "chapter-and-verse-20260930-182426-before-redo.dump"


def touch(directory, *names):
    for n in names:
        (directory / n).write_bytes(b"x")


def test_list_backups_ignores_other_files(tmp_path):
    touch(tmp_path, "chapter-and-verse-20260102-000000.dump", "chapter-and-verse-20260101-000000-label.dump",
          "notes.txt", "chapter-and-verse-20260103-000000.dump.partial", "other-20260101-000000.dump")
    assert [p.name for p in B.list_backups(tmp_path)] == [
        "chapter-and-verse-20260101-000000-label.dump",
        "chapter-and-verse-20260102-000000.dump",
    ]


def test_list_backups_of_a_missing_directory_is_empty(tmp_path):
    assert B.list_backups(tmp_path / "nope") == []


def test_prune_keeps_the_newest(tmp_path):
    names = [f"chapter-and-verse-2026010{d}-000000.dump" for d in range(1, 6)]
    touch(tmp_path, *names, "unrelated.dump")
    removed = B.prune(tmp_path, keep=2)
    assert [p.name for p in removed] == names[:3]
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([*names[3:], "unrelated.dump"])


def test_prune_with_fewer_backups_than_keep_removes_nothing(tmp_path):
    touch(tmp_path, "chapter-and-verse-20260101-000000.dump")
    assert B.prune(tmp_path, keep=5) == []


def test_prune_refuses_to_delete_everything(tmp_path):
    with pytest.raises(ValueError):
        B.prune(tmp_path, keep=0)
