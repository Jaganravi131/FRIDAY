"""CLI helpers are package code, never code loaded from the user's data root."""
import sys

from friday import cli


def test_needle_does_not_modify_module_search_path(root, capsys):
    before = list(sys.path)
    try:
        assert cli.main(["needle", "--length", "128", "--depths", "32",
                         "--trials", "1", "--dataset-only"]) == 0
        assert sys.path == before
    finally:
        sys.path[:] = before
