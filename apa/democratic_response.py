"""
Democratic inference pipeline.

This module provides:
- DemocraticInference: Main orchestrator for democratic voting
- VoterPool: Collection of user reward models
- UserVoter: Individual voter scoring and ranking
- Response generation using base LLM

CLI Usage:
    python -m apa.democratic_response --query "What is AI?"
    python -m apa.democratic_response --interactive
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# LLM Loading and Response Generation
# =============================================================================

# Global cache for inference model
_MODEL = None
_TOKENIZER = None
_MODEL_NAME = None


def load_inference_llm(
    model_name: str | None = None,
    device_map: str = "auto",
    cache_dir: str | None = None,
) -> Tuple[Any, Any]:
    """Load the base LLM for response generation (cached)."""
    global _MODEL, _TOKENIZER, _MODEL_NAME
    from apa.config import configure_environment, HF_CACHE_DIR

    default_model = "Qwen/Qwen2.5-7B-Instruct"
    if model_name is None:
        model_name = default_model

    if _MODEL is not None and _MODEL_NAME == model_name:
        return _MODEL, _TOKENIZER

    configure_environment()
    if cache_dir is None:
        cache_dir = str(HF_CACHE_DIR)

    print(f"Loading inference LLM: {model_name}")

    _TOKENIZER = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device_map,
        trust_remote_code=True, cache_dir=cache_dir,
    )
    _MODEL_NAME = model_name

    print("Inference LLM loaded successfully.")
    return _MODEL, _TOKENIZER


def generate_responses(
    query: str,
    k: int = 5,
    model: Any | None = None,
    tokenizer: Any | None = None,
    temperature: float = 1.2,
    max_new_tokens: int = 512,
) -> list[str]:
    """Generate k diverse responses using temperature sampling."""
    if model is None or tokenizer is None:
        model, tokenizer = load_inference_llm()

    messages = [{"role": "user", "content": query}]
    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"User: {query}\n\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    responses = []
    for _ in range(k):
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                do_sample=True, top_p=0.95, pad_token_id=tokenizer.eos_token_id,
            )
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text[len(prompt):].strip()
        responses.append(response)

    return responses


# =============================================================================
# Voter Classes
# =============================================================================

class UserVoter:
    """A voter that scores responses based on a learned user preference model."""

    def __init__(
        self,
        user_id: str,
        user_vector: torch.Tensor,
        basis_matrix: torch.Tensor,
        metadata: dict | None = None,
    ):
        self.user_id = user_id
        self.w = user_vector
        self.V = basis_matrix
        self.metadata = metadata or {}
        self.reward_direction = self.V @ self.w

    def score_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Score response embeddings."""
        return embeddings @ self.reward_direction

    def rank_embeddings(self, embeddings: torch.Tensor) -> list[int]:
        """Rank response embeddings by score (highest first)."""
        scores = self.score_embeddings(embeddings)
        return torch.argsort(scores, descending=True).tolist()


class VoterPool:
    """A pool of user voters for democratic voting."""

    def __init__(self, basis_matrix: torch.Tensor):
        self.V = basis_matrix
        self.voters: dict[str, UserVoter] = {}
        self.embedding_model = None
        self.embedding_tokenizer = None

    def add_voter(self, user_id: str, user_vector: torch.Tensor, metadata: dict | None = None) -> None:
        voter = UserVoter(user_id=user_id, user_vector=user_vector, basis_matrix=self.V, metadata=metadata)
        self.voters[user_id] = voter

    def load_prism_users(self, user_vectors_path: Path | str, user_mapping_path: Path | str | None = None) -> None:
        """Load PRISM user vectors from checkpoint."""
        import json

        W = torch.load(user_vectors_path, map_location='cpu')

        if user_mapping_path and Path(user_mapping_path).exists():
            with open(user_mapping_path, 'r') as f:
                user_to_idx = json.load(f)
            idx_to_user = {v: k for k, v in user_to_idx.items()}
        else:
            idx_to_user = {i: f"prism_user_{i}" for i in range(W.shape[0])}

        for idx in range(W.shape[0]):
            user_id = idx_to_user.get(idx, f"prism_user_{idx}")
            self.add_voter(user_id=user_id, user_vector=W[idx], metadata={'source': 'prism', 'idx': idx})

        print(f"Loaded {W.shape[0]} PRISM user voters")

    def load_historical_users(self, historical_dir: Path | str) -> None:
        """Load historical user vectors from directory."""
        historical_dir = Path(historical_dir)

        for path in historical_dir.glob("W_C*.pt"):
            checkpoint = torch.load(path, map_location='cpu')
            user_id = checkpoint.get('user_id', path.stem)
            w = checkpoint['w']

            self.add_voter(
                user_id=user_id, user_vector=w,
                metadata={
                    'source': 'historical',
                    'century': checkpoint.get('century'),
                    'profile': checkpoint.get('user_profile'),
                },
            )

        n_historical = sum(1 for v in self.voters.values() if v.metadata.get('source') == 'historical')
        print(f"Loaded {n_historical} historical user voters")

    def get_voter(self, user_id: str) -> UserVoter | None:
        return self.voters.get(user_id)

    def get_all_user_ids(self) -> list[str]:
        return list(self.voters.keys())

    def get_user_metadata(self) -> dict[str, dict]:
        return {uid: v.metadata for uid, v in self.voters.items()}

    def collect_rankings(self, embeddings: torch.Tensor, user_ids: list[str] | None = None) -> dict[str, list[int]]:
        """Collect rankings from multiple voters."""
        if user_ids is None:
            user_ids = list(self.voters.keys())

        rankings = {}
        for user_id in user_ids:
            voter = self.voters.get(user_id)
            if voter:
                rankings[user_id] = voter.rank_embeddings(embeddings)

        return rankings

    def embed_responses(self, responses: list[str], query: str | None = None) -> torch.Tensor:
        """Embed responses for scoring."""
        from apa.train_lore_bases import embed_texts, get_embedding_model

        if self.embedding_model is None:
            self.embedding_model, self.embedding_tokenizer = get_embedding_model()

        if query:
            texts = [f"{query}\n\n{r}" for r in responses]
        else:
            texts = responses

        embeddings = embed_texts(texts, model=self.embedding_model, tokenizer=self.embedding_tokenizer, show_progress=False)
        return torch.tensor(embeddings, dtype=torch.float32)

    @classmethod
    def from_checkpoint(
        cls,
        lore_checkpoint: Path | str,
        prism_users_path: Path | str | None = None,
        historical_dir: Path | str | None = None,
    ) -> "VoterPool":
        """Create a VoterPool from checkpoints."""
        from apa.train_lore_bases import LoReRewardModel

        lore_model = LoReRewardModel.load(str(lore_checkpoint), device='cpu')
        V = lore_model.V.data.clone()
        pool = cls(basis_matrix=V)

        if prism_users_path:
            checkpoint_dir = Path(prism_users_path).parent
            user_mapping = checkpoint_dir / "user_to_idx.json"
            pool.load_prism_users(prism_users_path, user_mapping)

        if historical_dir:
            pool.load_historical_users(historical_dir)

        return pool


# =============================================================================
# Democratic Inference
# =============================================================================

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
            f"---\n{self.winner_response}"
        )


class DemocraticInference:
    """Democratic inference pipeline."""

    def __init__(
        self,
        voter_pool: VoterPool,
        k_responses: int = 5,
        m_voters: int = 10,
        model: Any = None,
        tokenizer: Any = None,
    ):
        self.voter_pool = voter_pool
        self.k_responses = k_responses
        self.m_voters = m_voters

        if model is None or tokenizer is None:
            model, tokenizer = load_inference_llm()
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, query: str, k: int | None = None, m: int | None = None) -> InferenceResult:
        from apa.levers.voter_sampling import random_sampling
        from apa.levers.voter_aggregation import borda_count

        k = k or self.k_responses
        m = m or self.m_voters

        print(f"Generating {k} responses...")
        responses = generate_responses(query, k, self.model, self.tokenizer)

        print("Embedding responses...")
        embeddings = self.voter_pool.embed_responses(responses, query=query)

        all_user_ids = self.voter_pool.get_all_user_ids()
        print(f"Sampling {m} voters from {len(all_user_ids)} available...")
        sampled_user_ids = random_sampling(all_user_ids, None, min(m, len(all_user_ids)), {})

        print("Collecting rankings...")
        rankings = self.voter_pool.collect_rankings(embeddings, sampled_user_ids)

        print("Aggregating rankings...")
        aggregate_ranking = borda_count(rankings, {})

        winner_idx = aggregate_ranking[0]
        winner_response = responses[winner_idx]

        return InferenceResult(
            query=query, responses=responses, rankings=rankings,
            aggregate_ranking=aggregate_ranking, winner_idx=winner_idx,
            winner_response=winner_response, sampled_user_ids=sampled_user_ids,
        )

    def run_interactive(self) -> None:
        """Run interactive inference loop."""
        print("\n" + "="*60)
        print("Democratic Inference - Interactive Mode")
        print("="*60)
        print(f"Voters available: {len(self.voter_pool.get_all_user_ids())}")
        print(f"K responses: {self.k_responses}")
        print(f"M voters: {self.m_voters}")
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
        k_responses: int = 5,
        m_voters: int = 10,
    ) -> "DemocraticInference":
        """Create DemocraticInference from checkpoints."""
        voter_pool = VoterPool.from_checkpoint(
            lore_checkpoint=lore_checkpoint,
            prism_users_path=prism_users_path,
            historical_dir=historical_dir,
        )
        return cls(voter_pool=voter_pool, k_responses=k_responses, m_voters=m_voters)


def quick_inference(
    query: str,
    lore_checkpoint: Path | str,
    prism_users_path: Path | str | None = None,
    historical_dir: Path | str | None = None,
    k: int = 5,
    m: int = 10,
) -> str:
    """Quick function for running democratic inference."""
    inference = DemocraticInference.from_checkpoints(
        lore_checkpoint=lore_checkpoint,
        prism_users_path=prism_users_path,
        historical_dir=historical_dir,
        k_responses=k, m_voters=m,
    )
    result = inference(query)
    return result.winner_response


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    """CLI entry point for democratic inference."""
    from apa.config import configure_environment, MODELS_DIR

    parser = argparse.ArgumentParser(
        description="Run democratic inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query", type=str, default=None, help="Query to run inference on")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--k", type=int, default=5, help="Number of responses to generate")
    parser.add_argument("--m", type=int, default=10, help="Number of voters to sample")
    parser.add_argument("--lore_checkpoint", type=str, default=None, help="Path to LoRe model checkpoint")
    parser.add_argument("--prism_users", type=str, default=None, help="Path to PRISM user vectors")
    parser.add_argument("--historical_dir", type=str, default=None, help="Directory with historical user vectors")
    parser.add_argument("--show_all", action="store_true", help="Show all responses and rankings")
    args = parser.parse_args()

    if not args.query and not args.interactive:
        print("Error: Either --query or --interactive is required")
        sys.exit(1)

    configure_environment()

    lore_checkpoint = Path(args.lore_checkpoint) if args.lore_checkpoint else MODELS_DIR / "V_K8.pt"
    prism_users = Path(args.prism_users) if args.prism_users else MODELS_DIR / "W_seen_K8.pt"
    historical_dir = Path(args.historical_dir) if args.historical_dir else MODELS_DIR

    if not lore_checkpoint.exists():
        print(f"ERROR: LoRe checkpoint not found: {lore_checkpoint}")
        print("Please train LoRe first: python -m apa.train_lore_bases")
        sys.exit(1)

    print("\n" + "="*60)
    print("Democratic Inference")
    print("="*60)
    print(f"LoRe checkpoint: {lore_checkpoint}")
    print(f"PRISM users: {prism_users if prism_users.exists() else 'Not found'}")
    print(f"Historical users: {historical_dir if historical_dir.exists() else 'Not found'}")
    print(f"K responses: {args.k}")
    print(f"M voters: {args.m}")
    print("="*60 + "\n")

    inference = DemocraticInference.from_checkpoints(
        lore_checkpoint=lore_checkpoint,
        prism_users_path=prism_users if prism_users.exists() else None,
        historical_dir=historical_dir if historical_dir.exists() else None,
        k_responses=args.k, m_voters=args.m,
    )

    print(f"Total voters: {len(inference.voter_pool.get_all_user_ids())}\n")

    if args.interactive:
        inference.run_interactive()
    else:
        print(f"Query: {args.query}\n")
        print("Running democratic inference...")
        result = inference(args.query)

        print("\n" + "="*60)
        print("RESULT")
        print("="*60)

        if args.show_all:
            print("\nAll generated responses:")
            for i, resp in enumerate(result.responses):
                print(f"\n--- Response {i+1} ---")
                print(resp[:500] + "..." if len(resp) > 500 else resp)

            print("\n\nRankings from sampled voters:")
            for user_id, ranking in result.rankings.items():
                print(f"  {user_id}: {ranking}")

            print(f"\nAggregate ranking: {result.aggregate_ranking}")

        print(f"\n{'='*60}")
        print(f"WINNER: Response #{result.winner_idx + 1}")
        print(f"{'='*60}")
        print(result.winner_response)
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
