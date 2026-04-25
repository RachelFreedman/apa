"""
Unit tests for preference generation utilities.
"""

import pytest

from apa.synthetic_prefs.historical_prefs import (
    parse_model_response,
    preference_from_logprobs,
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


class TestPreferenceFromLogprobs:
    """Tests for preference_from_logprobs combiner."""

    def test_both_directions_pick_physical_1(self):
        # Original: P("1")=0.9 → physical-1 wins.
        # Reversed: P("2")=0.85 → physical-1 wins (Option 2 in reversed = physical 1).
        out = preference_from_logprobs(0.9, 0.1, 0.15, 0.85)
        assert out["final_preference"] == "1"
        assert out["consistency"] == 1.0
        assert out["soft_preference_1"] > 0.5

    def test_both_directions_pick_physical_2(self):
        out = preference_from_logprobs(0.1, 0.9, 0.85, 0.15)
        assert out["final_preference"] == "2"
        assert out["consistency"] == 1.0
        assert out["soft_preference_1"] < 0.5

    def test_position_bias_disagreement(self):
        # Original picks 1 (physical-1), reversed picks 1 (physical-2 in reversed) → disagree.
        out = preference_from_logprobs(0.8, 0.2, 0.8, 0.2)
        assert out["final_preference"] == "-1"
        assert out["consistency"] == 0.0

    def test_soft_preference_average(self):
        # Symmetric strong signal for physical 1.
        out = preference_from_logprobs(0.7, 0.3, 0.3, 0.7)
        # p1_phys = 0.5*(0.7+0.7)=0.7, p2_phys = 0.5*(0.3+0.3)=0.3 → soft_pref_1 = 0.7.
        assert abs(out["soft_preference_1"] - 0.7) < 1e-9

    def test_missing_token_treated_as_zero(self):
        # Caller passes 0.0 for a token that wasn't in the top-k logprobs.
        out = preference_from_logprobs(1.0, 0.0, 0.0, 1.0)
        assert out["final_preference"] == "1"
        assert out["consistency"] == 1.0
        assert out["soft_preference_1"] == 1.0

    def test_zero_total_returns_neutral_soft(self):
        out = preference_from_logprobs(0.0, 0.0, 0.0, 0.0)
        # When there's no signal at all (degenerate), soft pref defaults to 0.5.
        assert out["soft_preference_1"] == 0.5

    def test_echoes_input_probs(self):
        out = preference_from_logprobs(0.6, 0.4, 0.3, 0.7)
        assert out["prob_1_original"] == 0.6
        assert out["prob_2_original"] == 0.4
        assert out["prob_1_reversed"] == 0.3
        assert out["prob_2_reversed"] == 0.7
