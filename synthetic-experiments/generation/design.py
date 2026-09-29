"""Design of the synthetic value space: dimensions, questions, target profiles.

Everything the generator and rater prompts are built from lives here, so the
prompts in ../prompts/ can be regenerated exactly (python generation/design.py).
"""
import json, os

# MFQ-2 foundations, definitions verbatim from Atari et al. (2023), Table 2
# (J. Pers. Soc. Psychol., doi:10.1037/pspp0000470).
FOUNDATIONS = [
    ("Care", "Intuitions about avoiding emotional and physical damage to another individual."),
    ("Equality", "Intuitions about equal treatment and equal outcome for individuals."),
    ("Proportionality", "Intuitions about individuals getting rewarded in proportion to their merit or contribution."),
    ("Loyalty", "Intuitions about cooperating with ingroups and competing with outgroups."),
    ("Authority", "Intuitions about deference toward legitimate authorities and the defense of traditions, "
                  "all of which are seen as providing stability and fending off chaos."),
    ("Purity", "Intuitions about avoiding bodily and spiritual contamination and degradation."),
]
DIMS = [f for f, _ in FOUNDATIONS]

# Demo questions (kept from the paper) with designed target profiles.
# Order of each profile: (Care, Equality, Proportionality, Loyalty, Authority, Purity), each in {-2..2}.
# Identical profiles within a question are deliberate near-clones (to be paraphrased, not copied).
DEMO = {
    "Q1": dict(question="Should women have the same legal and political rights as men?", targets=[
        ("egalitarian rights (A)",          ( 1,  2,  0,  0, -1,  0)),
        ("egalitarian rights (A, clone)",   ( 1,  2,  0,  0, -1,  0)),
        ("substantive equality",            ( 2,  2, -1,  0, -1,  0)),
        ("divinely ordained spheres (B)",   ( 0, -2,  1,  1,  2,  2)),
        ("divinely ordained spheres (B, clone)", ( 0, -2,  1,  1,  2,  2)),
        ("divinely ordained spheres (B, clone 2)", ( 0, -2,  1,  1,  2,  2)),
        ("rights proportionate to station", ( 0, -2,  2,  0,  1,  0)),
        ("civil but not political rights",  ( 1,  0,  1,  1,  1,  0)),
        ("protective paternalism",          ( 2, -1,  0,  1,  1,  1)),
        ("communitarian skepticism of rights talk", ( 1,  0,  0,  2,  1,  0)),
    ]),
    "Q2": dict(question="Should marriage be defined as a lifelong union between a man and a woman in which the husband is the head of the household?", targets=[
        ("partnership of equals (A)",        ( 1,  2,  0,  0, -2, -1)),
        ("partnership of equals (A, clone)", ( 1,  2,  0,  0, -2, -1)),
        ("holy matrimony, husband as head (B)",        ( 0, -2,  0,  1,  2,  2)),
        ("holy matrimony, husband as head (B, clone)", ( 0, -2,  0,  1,  2,  2)),
        ("private contract, not for the state",        ( 0,  1,  1, -1, -2, -1)),
        ("soft complementarity: headship as duty of provision", ( 1, -1,  1,  1,  1,  1)),
        ("lifelong commitment for children's sake, no headship", ( 2,  1,  0,  2,  0,  0)),
        ("natural-law procreative ends",     ( 1, -1,  0,  1,  1,  2)),
        ("the formula has harmed women and others", ( 2,  2,  0, -1, -1, -2)),
        ("household order as the order of the realm", ( 0, -2,  1,  2,  2,  0)),
    ]),
    "Q3": dict(question="Should those accused of witchcraft be tried and punished by the state?", targets=[
        ("yes, the magistrate must root it out (B)",          ( 1,  0,  1,  1,  2,  2)),
        ("yes, the magistrate must root it out (B, clone)",   ( 1,  0,  1,  1,  2,  2)),
        ("yes, the magistrate must root it out (B, clone 2)", ( 1,  0,  1,  1,  2,  2)),
        ("no, superstition that kills innocents (A)",         ( 2,  1,  0,  0, -1, -1)),
        ("no, superstition that kills innocents (A, clone)",  ( 2,  1,  0,  0, -1, -1)),
        ("prosecute only real underlying crimes",             ( 2,  0,  2,  0,  1, -1)),
        ("a matter for church penance, not the civil sword",  ( 1,  0,  0,  1,  1,  2)),
        ("medical and psychological care instead",            ( 2,  0, -1,  0,  0, -2)),
        ("due process and liberty of conscience forbid it",   ( 1,  2,  0,  0, -2, -2)),
        ("punish to protect the community, mercy for the penitent", ( 1,  0,  1,  2,  2,  1)),
    ]),
}

# Calibration questions for Stage-3 fitting (no designed targets; rated blind like everything else).
CALIB = {
    "C1": "Should blasphemy be punishable by law?",
    "C2": "Should children be taught to obey their parents without question?",
    "C3": "Should the state redistribute wealth to reduce inequality between rich and poor?",
    "C4": "Should a person's first loyalty be to their own nation and people rather than to humanity at large?",
    "C5": "Should those who commit serious crimes be punished severely, even when it does no one any good?",
    "C6": "Should a married couple be permitted to divorce?",
}
N_CALIB = 8

def foundation_block():
    return "\n".join(f"- {name}: {d}" for name, d in FOUNDATIONS)

def profile_str(p):
    return ", ".join(f"{d} {v:+d}" for d, v in zip(DIMS, p))

GEN_DEMO_PROMPT = """You are helping construct a clearly-labelled SYNTHETIC dataset for a research paper on aggregating diverse and historically changing moral views with voting rules. Your job is to write candidate answers to a question. The answers will be presented as synthetic in the paper.

We describe moral outlooks using the six foundations of the Moral Foundations Questionnaire-2 (Atari et al., 2023):
{foundations}

For each target below, write ONE candidate answer to the question whose stance and reasoning match the target profile. A profile gives, for each foundation, how strongly the answer should uphold (+2 strongly, +1 somewhat) or go against / subordinate (-1 somewhat, -2 strongly) that concern, or leave it unengaged (0).

Requirements:
- 35 to 60 words each, written as a direct answer to the question, in the voice of someone sincerely holding that view. Views should be ones that people have plausibly held at some point between the 16th and 21st centuries.
- Do NOT name the foundations or use the words care, equality, proportionality, loyalty, authority, or purity (or their obvious derivatives such as "equal", "loyal", "pure", "authoritative"); convey the stance through content and reasoning instead. (Other people will later rate your answers on these dimensions without seeing the targets.)
- Where two or more targets have identical profiles, write genuinely different wordings of essentially the same position (a paraphrase with different phrasing and examples, not a copy).
- No slurs, no calls to violence against real groups; historical views may be illiberal but should be expressed as their holders would have argued them.

Question: "{question}"

Targets:
{targets}

Return ONLY a JSON list, with no other text, of the form
[{{"target_id": 1, "text": "..."}}, ...]
with one entry per target, in order.

Answer directly, without using any tools."""

GEN_CALIB_PROMPT = """You are helping construct a clearly-labelled SYNTHETIC dataset for a research paper on aggregating diverse and historically changing moral views with voting rules. Your job is to write candidate answers to several questions. The answers will be presented as synthetic in the paper.

We describe moral outlooks using the six foundations of the Moral Foundations Questionnaire-2 (Atari et al., 2023):
{foundations}

For EACH question below, write {n} different candidate answers that together span the range of positions people have plausibly held between the 16th and 21st centuries. Vary which of the six concerns each answer upholds or goes against, so that across the {n} answers to a question, as many of the six concerns as plausible are both upheld by some answer and rejected or subordinated by another.

Requirements:
- 35 to 60 words each, written as a direct answer to the question, in the voice of someone sincerely holding that view.
- Do NOT name the foundations or use the words care, equality, proportionality, loyalty, authority, or purity (or their obvious derivatives such as "equal", "loyal", "pure", "authoritative"); convey the stance through content and reasoning instead. (Other people will later rate your answers on these dimensions.)
- No slurs, no calls to violence against real groups; historical views may be illiberal but should be expressed as their holders would have argued them.

Questions:
{questions}

Return ONLY a JSON object, with no other text, mapping each question id to a list of {n} strings:
{{"C1": ["...", ...], "C2": [...], ...}}

Answer directly, without using any tools."""

RATE_PROMPT = """You are an expert annotator for a research study in moral psychology. You will rate short answers to moral questions on the six foundations of the Moral Foundations Questionnaire-2 (Atari et al., 2023):
{foundations}

For each answer, rate how strongly the answer's STANCE AND REASONING, with respect to the question it answers, upholds or goes against each foundation, on this signed scale:
  +2 = strongly upholds or relies on this concern
  +1 = somewhat upholds it
   0 = does not engage this concern, or is neutral about it
  -1 = somewhat goes against, dismisses, or subordinates it
  -2 = strongly goes against or rejects it
Rate each answer independently of the others, and rate each foundation independently of the other foundations. Judge what the answer says and implies, not whether you agree with it. Answers are presented in random order.

Answers:
{items}

Return ONLY a JSON object, with no other text, mapping each answer id to a list of six integers in the order [Care, Equality, Proportionality, Loyalty, Authority, Purity], e.g. {{"R01": [1, 0, -1, 0, 2, 0], ...}}. Include every answer id exactly once.

Answer directly, without using any tools."""

if __name__ == '__main__':
    # Writes the generation prompts to ../prompts and the design (foundations, questions, targets) to ../data.
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    P, D = os.path.join(ROOT, 'prompts'), os.path.join(ROOT, 'data')
    os.makedirs(P, exist_ok=True); os.makedirs(D, exist_ok=True)
    for qid, q in DEMO.items():
        targets = "\n".join(f"{i+1}. {profile_str(p)}" for i, (_, p) in enumerate(q['targets']))
        open(os.path.join(P, f'generate_{qid}.txt'), 'w').write(
            GEN_DEMO_PROMPT.format(foundations=foundation_block(), question=q['question'], targets=targets))
    qs = "\n".join(f'{cid}: "{q}"' for cid, q in CALIB.items())
    open(os.path.join(P, 'generate_calibration.txt'), 'w').write(
        GEN_CALIB_PROMPT.format(foundations=foundation_block(), n=N_CALIB, questions=qs))
    json.dump(dict(dims=DIMS, foundations=FOUNDATIONS, demo=DEMO, calib=CALIB), open(os.path.join(D, 'design.json'), 'w'), indent=1)
