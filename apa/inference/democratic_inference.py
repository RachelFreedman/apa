"""
Democratic Inference Pipeline.

The main orchestrator that:
1. Generates diverse responses using the base LLM
2. Samples user models from the voter pool
3. Collects rankings from each user
4. Aggregates rankings democratically
5. Returns the winning response
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from apa.config import APAConfig, InferenceConfig
from apa.inference.response_generator import generate_responses, load_inference_llm
from apa.inference.voter import VoterPool
from apa.levers import (
    lever_generate_responses,
    lever_sample_users,
    lever_aggregate_rankings,
)


@dataclass
class InferenceResult:
    """Result from democratic inference."""

    query: str
    responses: list[str]
    rankings: dict[str, list[int]]
    aggregate_ranking: list[int]
    winner_idx: int
    winner_response: str
    sampled_user_ids: list[str]

    def __str__(self) -> str:
        return (
            f"Query: {self.query[:100]}...\n"
            f"Generated {len(self.responses)} responses\n"
            f"Sampled {len(self.sampled_user_ids)} voters\n"
            f"Winner: Response #{self.winner_idx + 1}\n"
            f"---\n"
            f"{self.winner_response}"
        )


class DemocraticInference:
    """
    Democratic inference pipeline.

    Orchestrates the full process of generating responses,
    collecting votes, and aggregating preferences.
    """

    def __init__(
        self,
        voter_pool: VoterPool,
        config: InferenceConfig | None = None,
        model: Any = None,
        tokenizer: Any = None,
    ):
        """
        Initialize democratic inference.

        Args:
            voter_pool: Pool of user voters
            config: Inference configuration
            model: Pre-loaded LLM (loads default if None)
            tokenizer: Pre-loaded tokenizer (loads default if None)
        """
        self.voter_pool = voter_pool
        self.config = config or InferenceConfig()

        if model is None or tokenizer is None:
            model, tokenizer = load_inference_llm()

        self.model = model
        self.tokenizer = tokenizer

    def __call__(
        self,
        query: str,
        k: int | None = None,
        m: int | None = None,
    ) -> InferenceResult:
        """
        Run democratic inference on a query.

        Args:
            query: User query to respond to
            k: Number of responses to generate (uses config default if None)
            m: Number of voters to sample (uses config default if None)

        Returns:
            InferenceResult with winner and full details
        """
        k = k or self.config.k_responses
        m = m or self.config.m_voters

        # Build lever config
        lever_config = {
            'generate': self.config.generate_strategy,
            'sample': self.config.sample_strategy,
            'aggregate': self.config.aggregate_strategy,
            'temperature': 1.2,
            'max_new_tokens': 512,
        }

        # 1. Generate k diverse responses
        print(f"Generating {k} responses...")
        responses = lever_generate_responses(
            self.model,
            self.tokenizer,
            query,
            k,
            lever_config,
        )

        # 2. Embed responses
        print("Embedding responses...")
        embeddings = self.voter_pool.embed_responses(responses, query=query)

        # 3. Sample m voters
        all_user_ids = self.voter_pool.get_all_user_ids()
        user_metadata = self.voter_pool.get_user_metadata()

        print(f"Sampling {m} voters from {len(all_user_ids)} available...")
        sampled_user_ids = lever_sample_users(
            all_user_ids,
            user_metadata,
            m,
            lever_config,
        )

        # 4. Collect rankings from each voter
        print("Collecting rankings...")
        rankings = self.voter_pool.collect_rankings(embeddings, sampled_user_ids)

        # 5. Aggregate rankings
        print("Aggregating rankings...")
        aggregate_ranking = lever_aggregate_rankings(rankings, lever_config)

        # 6. Get winner
        winner_idx = aggregate_ranking[0]
        winner_response = responses[winner_idx]

        return InferenceResult(
            query=query,
            responses=responses,
            rankings=rankings,
            aggregate_ranking=aggregate_ranking,
            winner_idx=winner_idx,
            winner_response=winner_response,
            sampled_user_ids=sampled_user_ids,
        )

    def run_interactive(self) -> None:
        """
        Run interactive inference loop.

        Prompts for queries and displays results until user quits.
        """
        print("\n" + "="*60)
        print("Democratic Inference - Interactive Mode")
        print("="*60)
        print(f"Voters available: {len(self.voter_pool.get_all_user_ids())}")
        print(f"K responses: {self.config.k_responses}")
        print(f"M voters: {self.config.m_voters}")
        print("Type 'quit' to exit\n")

        while True:
            try:
                query = input("Query: ").strip()
            except EOFError:
                break

            if query.lower() in ['quit', 'exit', 'q']:
                break

            if not query:
                continue

            result = self(query)
            print("\n" + str(result) + "\n")

    @classmethod
    def from_checkpoints(
        cls,
        lore_checkpoint: Path | str,
        prism_users_path: Path | str | None = None,
        historical_dir: Path | str | None = None,
        config: InferenceConfig | None = None,
    ) -> "DemocraticInference":
        """
        Create DemocraticInference from checkpoints.

        Args:
            lore_checkpoint: Path to LoRe model checkpoint
            prism_users_path: Optional path to PRISM user vectors
            historical_dir: Optional directory with historical user vectors
            config: Inference configuration

        Returns:
            Initialized DemocraticInference
        """
        voter_pool = VoterPool.from_checkpoint(
            lore_checkpoint=lore_checkpoint,
            prism_users_path=prism_users_path,
            historical_dir=historical_dir,
        )

        return cls(voter_pool=voter_pool, config=config)


def quick_inference(
    query: str,
    lore_checkpoint: Path | str,
    prism_users_path: Path | str | None = None,
    historical_dir: Path | str | None = None,
    k: int = 5,
    m: int = 10,
) -> str:
    """
    Quick function for running democratic inference.

    Args:
        query: Query to respond to
        lore_checkpoint: Path to LoRe checkpoint
        prism_users_path: Optional PRISM users path
        historical_dir: Optional historical users directory
        k: Number of responses
        m: Number of voters

    Returns:
        Winning response string
    """
    config = InferenceConfig(k_responses=k, m_voters=m)

    inference = DemocraticInference.from_checkpoints(
        lore_checkpoint=lore_checkpoint,
        prism_users_path=prism_users_path,
        historical_dir=historical_dir,
        config=config,
    )

    result = inference(query)
    return result.winner_response
