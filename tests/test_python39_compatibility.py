"""Prevent accidental reintroduction of Python >3.9-only source constructs."""

from pathlib import Path

from tools.check_python39_compat import check_file, python_files


def test_project_source_is_python39_compatible():
    errors = []
    for path in python_files(Path(__file__).resolve().parents[1]):
        errors.extend(check_file(path))
    assert errors == [], "\n" + "\n".join(errors)
