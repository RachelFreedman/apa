"""Table and figures for Section 5, from outputs/results.json. Writes to outputs/figs/:
  table_votes.tex   Table tab:synth_votes   winners by question, jury and rule
  fig_jurors.pdf    Figure fig:jury         example jurors: true vs fitted weights
  fig_values.pdf    Figure fig:values       each period's weights beside the features of the response its jury elects
  winner_grid.pdf   Figure fig:winners      where all four rules agree (also .png)
"""
import json, os
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, FancyBboxPatch
from world import sample_cohort, fit_jurors, slate, DIMS, DEMO_Q, COHORTS, OUT

R = json.load(open(os.path.join(OUT, 'results.json')))
FIGS = os.path.join(OUT, 'figs'); os.makedirs(FIGS, exist_ok=True)
save = lambda fig, name, png_dpi=300: (fig.savefig(os.path.join(FIGS, name + '.pdf')),
                                       fig.savefig(os.path.join(FIGS, name + '.png'), dpi=png_dpi), plt.close(fig))
RULE_ORDER = ['Plurality', 'IRV-PUT', 'Copeland', 'Borda']            # first-place votes -> full ranking
MODAL = {j: {q: {r: int(next(iter(c))) for r, c in dq.items()} for q, dq in d.items()} for j, d in R['votes'].items()}
QLAB = {'Q1': 'Q1: Gender Equality', 'Q2': 'Q2: Marriage', 'Q3': 'Q3: Witchcraft'}
# Design language: the paper's Figure 1 blue/magenta, warm-gray neutral, ink greys.
BLUE, MID, MAGENTA, INK, SUBINK, GRID = '#0072C6', '#F0EFEC', '#C8009E', '#333333', '#666666', '#E4E3E0'
CMAP = LinearSegmentedColormap.from_list('apa_div', [(0, BLUE), (0.25, '#7DB6E3'), (0.5, MID), (0.75, '#E58CCF'), (1, MAGENTA)])

# ---------------- Table: winners over 200 jury draws ----------------
rules = ['Plurality', 'Borda', 'IRV-PUT', 'Copeland']
lines = [r'\begin{table*}[t]\centering\small', r'\begin{tabular}{@{}llccccc@{}}\toprule',
         r'Question & Jury & $\bar\rho$ & Plurality & Borda & IRV-PUT & Copeland\\\midrule']
for q in DEMO_Q:
    for j in ['16C', '20C', '21C', 'Mixed (10 each)']:
        d = R['votes'][j][q]; cells = []
        for r in rules:
            c = d[r]; k = max(c, key=c.get); cells.append('\\#' + str(k) + f' ({c[k] / sum(c.values()):.2f})')
        rho = R['walkthrough_rho'][j][q]
        lines.append(f"{QLAB[q] if j == '16C' else ''} & {j.replace(' (10 each)', '')} & {rho:.2f} & " + ' & '.join(cells) + '\\\\')
    lines.append(r'\midrule' if q != 'Q3' else r'\bottomrule')
lines += [r'\end{tabular}',
          r'\caption{Winners of the synthetic vote by jury composition and voting rule (fitted weights, $m=20$ comparisons per juror; 30 jurors; mixed = 10 per period). Each cell gives the most frequent winner over 200 independent jury draws (each draw samples new jurors and re-fits their weights); the number in parentheses is the fraction of those 200 draws in which that response won (ties broken by lowest index). $\bar\rho$: mean pairwise Spearman correlation among jurors (one draw).}',
          r'\label{tab:synth_votes}', r'\end{table*}']
open(os.path.join(FIGS, 'table_votes.tex'), 'w').write('\n'.join(lines))

# ---------------- Figure: example jurors, true vs fitted weights ----------------
with plt.rc_context({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42}):
    rng = np.random.default_rng(7)
    groups = [('16th c.', '16C'), ('20th c.', '20C'), ('21st c.', '21C')]
    Wt = [sample_cohort(rng, c, 4) for _, c in groups]
    Wf = [fit_jurors(rng, W, 20) for W in Wt]
    vmax = np.ceil(max(np.abs(np.concatenate(Wt)).max(), np.abs(np.concatenate(Wf)).max()) * 10) / 10
    GAP = 0.45; tops = [g * (4 + GAP) for g in range(3)]
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 2.6), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, blocks, ttl in zip(axs, [Wt, Wf], [r'true $w$', r'fitted $\hat w$ ($m=20$)']):
        for top, Mx in zip(tops, blocks):
            for i in range(Mx.shape[0]):
                for j in range(Mx.shape[1]):
                    ax.add_patch(Rectangle((j, top + i), 1, 1, facecolor=CMAP((Mx[i, j] + vmax) / (2 * vmax)),
                                           edgecolor='white', linewidth=1.0))
        ax.set_xlim(0, 6); ax.set_ylim(tops[-1] + 4, 0); ax.set_xticks(np.arange(6) + 0.5)
        ax.set_xticklabels(DIMS, rotation=45, ha='right', rotation_mode='anchor', fontsize=6.5, color=INK)
        ax.tick_params(length=0, pad=2); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(ttl, fontsize=7.5, color=INK, pad=4)
    ax = axs[0]
    for (lab, _), top in zip(groups, tops):                               # century brackets
        y0, y1, x = top + 0.12, top + 3.88, -0.28
        ax.plot([x, x], [y0, y1], color=INK, lw=0.9, clip_on=False, solid_capstyle='butt')
        for y in (y0, y1): ax.plot([x, x + 0.14], [y, y], color=INK, lw=0.9, clip_on=False)
        ax.text(x - 0.18, top + 2, lab, ha='right', va='center', fontsize=7, color=INK, clip_on=False)
    cax = fig.add_axes([0.42, 0.075, 0.34, 0.028])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(-vmax, vmax)), cax=cax, orientation='horizontal')
    cb.set_ticks([-1.5, -0.75, 0, 0.75, 1.5]); cb.set_ticklabels(['−1.5', '', '0', '', '1.5'])
    cb.ax.tick_params(labelsize=6, length=2, color=SUBINK, labelcolor=SUBINK); cb.outline.set_visible(False)
    cax.text(-0.04, 0.5, 'weight', transform=cax.transAxes, ha='right', va='center', fontsize=6.5, color=SUBINK)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.91, bottom=0.34)
    save(fig, 'fig_jurors')

# ---------------- Figure: values -> elected responses (double column) ----------------
with plt.rc_context({'font.size': 7, 'pdf.fonttype': 42}):
    WMAX, FMAX = 1.5, 2.0
    ROWS = [('16C', '16th century'), ('20C', '20th century'), ('21C', '21st century')]
    RH, RGAP = 1.25, 0.95
    X0 = [0, 9.0, 16.9, 24.8]
    fig = plt.figure(figsize=(7.0, 2.3)); ax = fig.add_axes([0.0, 0.0, 0.87, 1.0])
    def strip(x, y, h, vec, vmax):
        for j, v in enumerate(vec):
            ax.add_patch(Rectangle((x + j, y), 1, h, fc=CMAP((v + vmax) / (2 * vmax)), ec='white', lw=1.0))
    for r, (ck, cl) in enumerate(ROWS):
        y = r * (RH + RGAP)
        ax.text(-1.75, y + RH / 2, cl, ha='right', va='center', fontsize=7.5, color=INK)
        yy = y
        for share, mu in COHORTS[ck]:                                     # mixture components, height = share
            strip(X0[0], yy, RH * share, mu, WMAX)
            ax.text(-0.15, yy + RH * share / 2, f'{share:.0%}', ha='right', va='center', fontsize=5.8, color=SUBINK)
            yy += RH * share
        ax.text(7.05, y + RH / 2, '→', ha='center', va='center', fontsize=11, color=SUBINK)
        for q, x in zip(DEMO_Q, X0[1:]):
            uniq = list(dict.fromkeys(MODAL[ck][q][rr] for rr in RULE_ORDER)); F = slate(q); h = RH / len(uniq)
            for n, resp in enumerate(uniq): strip(x, y + n * h, h, F[resp - 1], FMAX)
            ax.text(x - 0.25, y + (h / 2 if len(uniq) > 1 else RH / 2), f'#{uniq[0]}', ha='right', va='center', fontsize=6.8, color=INK)
            if len(uniq) > 1:
                ax.text(x - 0.25, y + 1.5 * h, f'#{uniq[1]}', ha='right', va='center', fontsize=6.8, color=INK)
    ytop = -0.45
    ax.text(X0[0] + 3, ytop, 'juror weights', ha='center', va='bottom', fontsize=7.5, color=INK)
    for q, x in zip(DEMO_Q, X0[1:]): ax.text(x + 3, ytop, QLAB[q], ha='center', va='bottom', fontsize=7.5, color=INK)
    ax.plot([X0[1] - 1.2, X0[3] + 6], [ytop - 1.25, ytop - 1.25], color='#CFCCC4', lw=0.7)
    ax.text((X0[1] - 1.2 + X0[3] + 6) / 2, ytop - 1.45, 'features of the response each jury elects', ha='center',
            va='bottom', fontsize=7, color=SUBINK)
    ybot = 3 * RH + 2 * RGAP + 0.25
    for x in X0:
        for j, t in enumerate(DIMS):
            ax.text(x + j + 0.62, ybot, t, ha='right', va='top', rotation=45, rotation_mode='anchor', fontsize=6.5, color=INK)
    ax.set_xlim(-6.5, X0[3] + 6.2); ax.set_ylim(ybot + 3.9, ytop - 2.5); ax.set_aspect('equal'); ax.axis('off')
    for n, (vmax, lab) in enumerate([(WMAX, 'weight'), (FMAX, 'feature')]):
        cax = fig.add_axes([0.89 + n * 0.057, 0.345, 0.011, 0.42])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=Normalize(-vmax, vmax)), cax=cax, orientation='vertical')
        cb.set_ticks([-vmax, 0, vmax]); cb.set_ticklabels([f'−{vmax:g}', '0', f'{vmax:g}'])
        cb.ax.tick_params(labelsize=6, length=1.5, color=SUBINK, labelcolor=SUBINK, pad=1.5); cb.outline.set_visible(False)
        cax.set_title(lab, fontsize=6.5, color=SUBINK, pad=4)
    save(fig, 'fig_values')

# ---------------- Figure: winner grid (do all four rules agree?) ----------------
with plt.rc_context({'font.size': 7, 'pdf.fonttype': 42}):
    JK = [('16C', '16th c.'), ('20C', '20th c.'), ('21C', '21st c.'), ('Mixed (10 each)', 'Mixed')]
    fig, ax = plt.subplots(figsize=(3.4, 2.25))
    GAPQ, H = 0.9, 0.72
    ypos = {'16C': 3.3, '20C': 2.3, '21C': 1.3, 'Mixed (10 each)': 0.0}
    for qi, q in enumerate(DEMO_Q):
        x0 = qi * (4 + GAPQ)
        ax.text(x0 + 2, 4.55, QLAB[q], ha='center', va='bottom', fontsize=6.8, color=INK)
        for k, r in enumerate(['Plu', 'IRV', 'Cop', 'Bor']):
            ax.text(x0 + k + 0.5, 4.12, r, ha='center', va='bottom', fontsize=5.2, color=SUBINK)
        for jk, jl in JK:
            w = [MODAL[jk][q][r] for r in RULE_ORDER]
            runs, s = [], 0                                               # merge adjacent rules with the same winner
            for k in range(1, 5):
                if k == 4 or w[k] != w[s]: runs.append((s, k, w[s])); s = k
            agree = len(runs) == 1
            for a, b, resp in runs:
                ax.add_patch(FancyBboxPatch((x0 + a + 0.06, ypos[jk] - H / 2), b - a - 0.12, H,
                             boxstyle='round,pad=0,rounding_size=0.12', lw=0, fc='#3F3F3D' if agree else '#DAD7CF', zorder=2))
                ax.text(x0 + (a + b) / 2, ypos[jk], f'#{resp}', ha='center', va='center', fontsize=6.5,
                        color='white' if agree else INK, zorder=3)
                if jk.startswith('Mixed'):                                # which single-period jury elected this response?
                    src = [nm for kk, nm in [('16C', '16th'), ('20C', '20th'), ('21C', '21st')]
                           if len(set(MODAL[kk][q].values())) == 1 and MODAL[kk][q]['Borda'] == resp]
                    if src: ax.text(x0 + (a + b) / 2, ypos[jk] - H / 2 - 0.12, '= ' + ' & '.join(src) + ' c.', ha='center',
                                    va='top', fontsize=5.2, color=SUBINK)
    for jk, jl in JK: ax.text(-0.25, ypos[jk], jl, ha='right', va='center', fontsize=6.8, color=INK)
    ax.plot([-1.6, 3 * 4 + 2 * GAPQ], [0.8, 0.8], color=GRID, lw=0.8, clip_on=False)
    ax.set_xlim(-1.7, 3 * 4 + 2 * GAPQ + 0.05); ax.set_ylim(-1.55, 5.0); ax.axis('off')
    for n, (fc, t) in enumerate([('#3F3F3D', 'all four rules pick the same response'), ('#DAD7CF', 'rules disagree')]):
        xk = 0.2 + n * 9.0
        ax.add_patch(FancyBboxPatch((xk, -1.45), 0.7, 0.34, boxstyle='round,pad=0,rounding_size=0.08', lw=0, fc=fc))
        ax.text(xk + 0.9, -1.28, t, ha='left', va='center', fontsize=6, color=INK)
    fig.tight_layout(pad=0.2)
    save(fig, 'winner_grid')
print('wrote', sorted(os.listdir(FIGS)))
