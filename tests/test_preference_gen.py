"""
Unit tests for preference generation utilities.
"""

import pytest

from apa.historical_prefs import (
    parse_model_response,
    preferences_to_labels,
)


class TestParseModelResponse:
    """Tests for parse_model_response function."""

    def test_empty_response(self):
        """Test empty response returns -1."""
        assert parse_model_response('') == '-1'
        assert parse_model_response('   ') == '-1'

    def test_option_1_patterns(self):
        """Test various patterns for option 1."""
        assert parse_model_response('Option 1') == '1'
        assert parse_model_response('option 1') == '1'
        assert parse_model_response('I prefer option 1') == '1'
        assert parse_model_response('Choice 1 is better') == '1'
        assert parse_model_response('Response 1') == '1'
        assert parse_model_response('The first option') == '1'
        assert parse_model_response('first response') == '1'
        assert parse_model_response('#1') == '1'
        assert parse_model_response('1st option') == '1'

    def test_option_2_patterns(self):
        """Test various patterns for option 2."""
        assert parse_model_response('Option 2') == '2'
        assert parse_model_response('option 2') == '2'
        assert parse_model_response('I prefer option 2') == '2'
        assert parse_model_response('Choice 2 is better') == '2'
        assert parse_model_response('Response 2') == '2'
        assert parse_model_response('The second option') == '2'
        assert parse_model_response('second response') == '2'
        assert parse_model_response('#2') == '2'
        assert parse_model_response('2nd option') == '2'

    def test_ambiguous_both(self):
        """Test ambiguous responses mentioning both."""
        assert parse_model_response('Both option 1 and option 2 are good') == '-1'
        assert parse_model_response('Option 1 is ok but option 2 is better') == '-1'

    def test_starts_with_digit(self):
        """Test responses starting with digit."""
        assert parse_model_response('1') == '1'
        assert parse_model_response('2') == '2'
        assert parse_model_response('1.') == '1'
        assert parse_model_response('2.') == '2'

    def test_short_responses(self):
        """Test very short responses."""
        assert parse_model_response('1') == '1'
        assert parse_model_response('2') == '2'
        assert parse_model_response('is 1') == '1'
        assert parse_model_response('is 2') == '2'

    def test_unparseable(self):
        """Test unparseable responses."""
        assert parse_model_response('I cannot decide') == '-1'
        assert parse_model_response('Both are equally good') == '-1'
        assert parse_model_response('Neither is appropriate') == '-1'


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
