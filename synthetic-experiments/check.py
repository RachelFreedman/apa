"""Compare outputs/results.json with the numbers reported in the paper. Exits non-zero on any mismatch."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'outputs', 'results.json')))
S = json.load(open(os.path.join(HERE, 'outputs', 'rating_agreement.json')))

# Table tab:synth_votes: rho-bar, then modal winner (share of 200 draws) under Plurality, Borda, IRV-PUT, Copeland.
TABLE = {
 ('Q1', '16C'): ('0.74', '#6 (1.00)', '#6 (1.00)', '#6 (1.00)', '#6 (1.00)'),
 ('Q1', '20C'): ('0.08', '#8 (0.67)', '#8 (0.97)', '#8 (0.81)', '#8 (0.94)'),
 ('Q1', '21C'): ('0.50', '#3 (1.00)', '#3 (0.55)', '#3 (0.95)', '#3 (0.93)'),
 ('Q1', 'Mixed'): ('0.04', '#6 (0.55)', '#8 (0.66)', '#6 (0.67)', '#8 (0.66)'),
 ('Q2', '16C'): ('0.73', '#4 (0.98)', '#4 (0.95)', '#4 (0.98)', '#4 (0.95)'),
 ('Q2', '20C'): ('0.09', '#6 (0.64)', '#6 (0.85)', '#6 (0.81)', '#6 (0.84)'),
 ('Q2', '21C'): ('0.49', '#9 (0.89)', '#1 (0.94)', '#9 (0.64)', '#9 (0.51)'),
 ('Q2', 'Mixed'): ('0.01', '#4 (0.47)', '#6 (0.70)', '#4 (0.41)', '#6 (0.80)'),
 ('Q3', '16C'): ('0.85', '#3 (0.91)', '#3 (0.99)', '#3 (0.98)', '#3 (0.98)'),
 ('Q3', '20C'): ('0.08', '#3 (0.57)', '#3 (0.79)', '#3 (0.71)', '#3 (0.57)'),
 ('Q3', '21C'): ('0.40', '#9 (0.97)', '#4 (0.98)', '#9 (0.81)', '#4 (0.59)'),
 ('Q3', 'Mixed'): ('0.06', '#3 (0.63)', '#3 (0.93)', '#3 (0.88)', '#3 (0.81)'),
}
RANKING_AGREEMENT = {'16C': '0.91', '20C': '0.86', '21C': '0.90'}      # Section 5, "ranking agreement"
RATERS = {'linear_weighted_kappa': '0.94', 'target_vs_rating_r': '0.94'} # Appendix, feature rating

bad = 0
def check(label, got, want):
    global bad
    ok = got == want; bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {label}: {got}" + ('' if ok else f"   (paper: {want})"))
for (q, j), want in TABLE.items():
    jk = 'Mixed (10 each)' if j == 'Mixed' else j
    cells = [f"{R['walkthrough_rho'][jk][q]:.2f}"]
    for r in ['Plurality', 'Borda', 'IRV-PUT', 'Copeland']:
        c = R['votes'][jk][q][r]; k = max(c, key=c.get); cells.append(f"#{k} ({c[k] / sum(c.values()):.2f})")
    check(f"table {q} {j:5s}", tuple(cells), want)
for c, want in RANKING_AGREEMENT.items(): check(f"ranking agreement {c}", f"{R['ranking_agreement'][c]:.2f}", want)
for k, want in RATERS.items(): check(k, f"{S[k]:.2f}", want)
print('\nall checks passed' if not bad else f'\n{bad} check(s) failed'); sys.exit(bool(bad))
