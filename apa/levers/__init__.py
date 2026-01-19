"""
Levers: Injection points for customizable behavior.

Each lever is a function that can be swapped out for different strategies.
The default implementations are simple baselines that work but can be
replaced with more sophisticated methods later.
"""

from apa.levers.lever_generate import lever_generate_responses
from apa.levers.lever_sample import lever_sample_users
from apa.levers.lever_aggregate import lever_aggregate_rankings
from apa.levers.lever_questions import lever_select_questions

__all__ = [
    "lever_generate_responses",
    "lever_sample_users",
    "lever_aggregate_rankings",
    "lever_select_questions",
]
