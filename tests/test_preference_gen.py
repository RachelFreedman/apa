"""
Unit tests for preference generation utilities.
"""

import pytest

from apa.historical_prefs import (
    Preference,
    parse_model_response,
    preferences_to_labels,
)


class TestParseModelResponse:
    """Tests for parse_model_response function."""

    def test_empty_response(self):
        """Test empty response returns -1."""
        assert parse_model_response('') == Preference.AMBIGUOUS
        assert parse_model_response('   ') == Preference.AMBIGUOUS

    def test_option_1_patterns(self):
        """Test various patterns for option 1."""
        assert parse_model_response('Option 1') == Preference.OPTION_1
        assert parse_model_response('option 1') == Preference.OPTION_1
        assert parse_model_response('I prefer option 1') == Preference.OPTION_1
        assert parse_model_response('Choice 1 is better') == Preference.OPTION_1
        assert parse_model_response('Response 1') == Preference.OPTION_1
        assert parse_model_response('The first option') == Preference.OPTION_1
        assert parse_model_response('first response') == Preference.OPTION_1
        assert parse_model_response('#1') == Preference.OPTION_1
        assert parse_model_response('1st option') == Preference.OPTION_1

    def test_option_2_patterns(self):
        """Test various patterns for option 2."""
        assert parse_model_response('Option 2') == Preference.OPTION_2
        assert parse_model_response('option 2') == Preference.OPTION_2
        assert parse_model_response('I prefer option 2') == Preference.OPTION_2
        assert parse_model_response('Choice 2 is better') == Preference.OPTION_2
        assert parse_model_response('Response 2') == Preference.OPTION_2
        assert parse_model_response('The second option') == Preference.OPTION_2
        assert parse_model_response('second response') == Preference.OPTION_2
        assert parse_model_response('#2') == Preference.OPTION_2
        assert parse_model_response('2nd option') == Preference.OPTION_2

    def test_ambiguous_both(self):
        """Test ambiguous responses mentioning both."""
        assert parse_model_response('Both option 1 and option 2 are good') == Preference.AMBIGUOUS
        assert parse_model_response('Option 1 is ok but option 2 is better') == Preference.AMBIGUOUS

    def test_starts_with_digit(self):
        """Test responses starting with digit."""
        assert parse_model_response('1') == Preference.OPTION_1
        assert parse_model_response('2') == Preference.OPTION_2
        assert parse_model_response('1.') == Preference.OPTION_1
        assert parse_model_response('2.') == Preference.OPTION_2

    def test_short_responses(self):
        """Test very short responses."""
        assert parse_model_response('1') == Preference.OPTION_1
        assert parse_model_response('2') == Preference.OPTION_2
        assert parse_model_response('is 1') == Preference.OPTION_1
        assert parse_model_response('is 2') == Preference.OPTION_2

    def test_unparseable(self):
        """Test unparseable responses."""
        assert parse_model_response('I cannot decide') == Preference.AMBIGUOUS
        assert parse_model_response('Both are equally good') == Preference.AMBIGUOUS
        assert parse_model_response('Neither is appropriate') == Preference.AMBIGUOUS


class TestPreferencesToLabels:
    """Tests for preferences_to_labels function."""

    def test_binary_labels(self):
        """Test converting to binary labels."""
        preferences = [
            {'final_preference': '1'},
            {'final_preference': '2'},
            {'final_preference': '1'},
            {'final_preference': '-1'},
        ]

        labels = preferences_to_labels(preferences, as_binary=True)

        assert labels == [0, 1, 0, -1]

    def test_non_binary_labels(self):
        """Test converting to non-binary labels."""
        preferences = [
            {'final_preference': '1'},
            {'final_preference': '2'},
            {'final_preference': '-1'},
        ]

        labels = preferences_to_labels(preferences, as_binary=False)

        assert labels == [1, 2, -1]

    def test_missing_preference(self):
        """Test handling missing final_preference key."""
        preferences = [
            {'final_preference': '1'},
            {},  # Missing
            {'other': 'data'},
        ]

        labels = preferences_to_labels(preferences, as_binary=True)

        assert labels == [0, -1, -1]

    def test_empty_list(self):
        """Test empty preference list."""
        labels = preferences_to_labels([], as_binary=True)
        assert labels == []

    def test_int_and_enum_values(self):
        """Test int / Preference final_preference values (new JSON format)."""
        preferences = [
            {'final_preference': 1},
            {'final_preference': 2},
            {'final_preference': Preference.OPTION_1},
            {'final_preference': -1},
        ]
        assert preferences_to_labels(preferences, as_binary=True) == [0, 1, 0, -1]
        assert preferences_to_labels(preferences, as_binary=False) == [1, 2, 1, -1]


class TestPreferenceEnum:
    """Tests for the Preference enum contract."""

    def test_parse_returns_preference(self):
        assert isinstance(parse_model_response('Option 1'), Preference)

    def test_int_values(self):
        assert int(Preference.OPTION_1) == 1
        assert int(Preference.OPTION_2) == 2
        assert int(Preference.AMBIGUOUS) == -1
