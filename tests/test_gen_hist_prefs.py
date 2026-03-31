"""
Unit tests for synthetic historical preference generation.

Tests pure functions only — no GPU or model loading required.
"""

import json
import tempfile
from pathlib import Path

import pytest

from apa.synthetic_prefs.historical_prefs import (
    VALID_CENTURIES,
    load_profiles,
    results_to_jsonl_records,
    write_jsonl,
)
from apa.synthetic_prefs.eval_prefs import load_prefs_jsonl


# ---------------------------------------------------------------------------
# load_profiles
# ---------------------------------------------------------------------------

class TestLoadProfiles:
    """Test loading user profiles from JSONL."""

    def test_bundled_profiles(self):
        """Bundled profiles.jsonl loads correctly with expected structure."""
        profiles = load_profiles()
        assert "C013" in profiles
        assert "C019" in profiles
        assert len(profiles["C013"]) == 5
        assert len(profiles["C019"]) == 5

    def test_all_profiles_nonempty(self):
        """Every profile is a non-empty string."""
        profiles = load_profiles()
        for century, profs in profiles.items():
            assert century in VALID_CENTURIES
            for p in profs:
                assert isinstance(p, str)
                assert len(p) > 0

    def test_custom_profiles_file(self, tmp_path):
        """Can load profiles from a custom JSONL file."""
        path = tmp_path / "custom.jsonl"
        path.write_text(
            '{"century": "C017", "profile": "A 17th-century Dutch painter."}\n'
            '{"century": "C017", "profile": "A 17th-century Ottoman merchant."}\n'
        )
        profiles = load_profiles(path)
        assert profiles == {
            "C017": [
                "A 17th-century Dutch painter.",
                "A 17th-century Ottoman merchant.",
            ]
        }

    def test_empty_file(self, tmp_path):
        """Empty file returns empty dict."""
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        profiles = load_profiles(path)
        assert profiles == {}


# ---------------------------------------------------------------------------
# results_to_jsonl_records
# ---------------------------------------------------------------------------

def _make_result(user_id, prompt, r1, r2, pref):
    return {
        "user_id": user_id,
        "prompt": prompt,
        "response_1": r1,
        "response_2": r2,
        "final_preference": pref,
        "consistency": 0.8,
    }


class TestResultsToJsonlRecords:
    """Test conversion from raw results to eval_prefs JSONL format."""

    def test_preference_1(self):
        """Preference '1' maps chosen=response_1, rejected=response_2."""
        results = [_make_result("u0", "Q?", "A", "B", "1")]
        records = results_to_jsonl_records(results)
        assert len(records) == 1
        assert records[0] == {
            "user_id": "u0", "prompt": "Q?", "chosen": "A", "rejected": "B",
        }

    def test_preference_2(self):
        """Preference '2' maps chosen=response_2, rejected=response_1."""
        results = [_make_result("u0", "Q?", "A", "B", "2")]
        records = results_to_jsonl_records(results)
        assert len(records) == 1
        assert records[0]["chosen"] == "B"
        assert records[0]["rejected"] == "A"

    def test_ambiguous_skipped(self):
        """Preference '-1' (ambiguous) is skipped."""
        results = [_make_result("u0", "Q?", "A", "B", "-1")]
        records = results_to_jsonl_records(results)
        assert len(records) == 0

    def test_mixed(self):
        """Only valid preferences make it through."""
        results = [
            _make_result("u0", "Q1", "A", "B", "1"),
            _make_result("u0", "Q2", "C", "D", "-1"),
            _make_result("u1", "Q3", "E", "F", "2"),
        ]
        records = results_to_jsonl_records(results)
        assert len(records) == 2
        assert records[0]["user_id"] == "u0"
        assert records[1]["user_id"] == "u1"

    def test_empty_input(self):
        """Empty input returns empty list."""
        assert results_to_jsonl_records([]) == []


# ---------------------------------------------------------------------------
# write_jsonl + round-trip with load_prefs_jsonl
# ---------------------------------------------------------------------------

class TestWriteJsonl:
    """Test JSONL writing and round-trip compatibility with eval_prefs."""

    def test_round_trip(self, tmp_path):
        """Written JSONL can be loaded by eval_prefs.load_prefs_jsonl."""
        records = [
            {"user_id": "u0", "prompt": "Q1?", "chosen": "A", "rejected": "B"},
            {"user_id": "u0", "prompt": "Q2?", "chosen": "C", "rejected": "D"},
            {"user_id": "u1", "prompt": "Q1?", "chosen": "E", "rejected": "F"},
        ]
        path = tmp_path / "test.jsonl"
        write_jsonl(records, path)

        loaded = load_prefs_jsonl(path)
        assert set(loaded.keys()) == {"u0", "u1"}
        assert len(loaded["u0"]) == 2
        assert len(loaded["u1"]) == 1
        assert loaded["u0"][0].prompt == "Q1?"
        assert loaded["u0"][0].chosen == "A"
        assert loaded["u0"][0].rejected == "B"

    def test_creates_parent_dirs(self, tmp_path):
        """write_jsonl creates parent directories if needed."""
        path = tmp_path / "sub" / "dir" / "test.jsonl"
        write_jsonl([{"user_id": "u", "prompt": "Q", "chosen": "A", "rejected": "B"}], path)
        assert path.exists()

    def test_empty_records(self, tmp_path):
        """Empty records list produces empty file."""
        path = tmp_path / "empty.jsonl"
        write_jsonl([], path)
        assert path.read_text() == ""
