# Synthetic proof-of-concept experiments

This folder reproduces the proof-of-concept experiments in Section 5 of the paper. The setup is fully synthetic, and no model is trained:

- **Reward bases.** Fixed, interpretable bases: the six MFQ-2 moral foundations.
- **Candidate responses.** LLM-generated, then blind-rated on those foundations.
- **Annotators.** Simulated from hand-specified 16th-, 20th- and 21st-century weight distributions.

Stage 3 fits each new annotator's weights from 20 simulated comparisons. Stage 2 aggregates the fitted jurors' rankings with four voting rules.

## Quick start

```bash
cd synthetic-experiments
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run.sh
```

`run.sh` takes about a minute. It writes everything to `outputs/`, then checks the numbers against the paper. It should end with `all checks passed`.

| Step | Script | Produces | Paper |
|---|---|---|---|
| 1 | `features.py` | `outputs/features.json`: each response's feature vector φ(i), the mean of the two blind ratings. Also rater-agreement statistics. | Appendix |
| 2 | `experiments.py` | `outputs/results.json`: winners over 200 jury draws, the ρ̄ column, and ranking agreement per period. | Table `tab:synth_votes`, Section 5 text |
| 3 | `figures.py` | `outputs/figs/table_votes.tex`, `fig_jurors.pdf`, `fig_values.pdf`, `winner_grid.pdf` (and `.png`) | Table `tab:synth_votes`, Figures `fig:jury`, `fig:values`, `fig:winners` |
| 4 | `check.py` | Pass or fail for each reported number | |

Every analysis uses a fixed seed. With the versions in `requirements.txt`, the numbers match the paper exactly, and the figures match it pixel for pixel.

## What's in the folder

- `world.py`: the model. It holds the cohort weight distributions, the Bradley–Terry comparison simulator, the ridge-regularized fit (Newton's method), and the four voting rules (plurality, Borda, IRV-PUT, Copeland, via `pref_voting`).
- `data/`: the frozen inputs.
  - `design.json`: foundations (with the MFQ-2 definitions), questions, and design target profiles.
  - `generated_Q{1,2,3}.json`: the demonstration responses.
  - `generated_calibration.json`: the calibration responses used for Stage-3 comparisons.
  - `ratings_rater{A,B}.json`: the two blind ratings.
  - `rating_key.json`: maps the shuffled answer IDs shown to the raters back to responses.
- `prompts/`: the exact prompts sent to the LLM. `generate_*.txt` are the generation prompts and `rate_all.txt` is the blind-rating prompt.
- `generation/`: the scripts that built those prompts.
  - `design.py`: design → generation prompts.
  - `build_rating.py`: generated responses → shuffled rating prompt and key.

  Running either one reproduces `prompts/` and `data/` byte for byte.

## The LLM steps (not re-run)

Response generation and blind rating each called an LLM once, so they can't be re-run bit for bit. Their outputs in `data/` are the frozen starting point for `run.sh`.

Each prompt in `prompts/` went to a fresh-context Claude subagent (`opus` alias, session model `claude-opus-5-5`): one call per generation prompt, and two independent calls on `rate_all.txt` for raters A and B. The paper's appendix gives the full procedure.

To regenerate from scratch:

1. Run `python generation/design.py`.
2. Send each `prompts/generate_*.txt` to a model, and save the replies in the format of `data/generated_*.json`.
3. Run `python generation/build_rating.py`.
4. Collect two independent ratings of `prompts/rate_all.txt` as `data/ratings_rater{A,B}.json`.
5. Run `bash run.sh`. The numbers will then differ from the paper's, and `check.py` will report that.
