# Thesis-asset learnings

What this skill has learned from the user. Read it before every use. Entries here override `SKILL.md` and `style.md`. Keep entries short, general and dated. Merge or replace entries rather than adding ones that contradict each other.

## Style

- (2026-11-21) The user's own specs for the first two figures (Ch. 2 decision diagram, Ch. 2 ROC illustration) asked for: Times New Roman 11 pt, black and grey only, no title, Greek decimal comma, real Unicode Ŷ, compact enough for the A4 text width. These are now the defaults in `style.md`.

## Terms and notation

- (2026-11-21) Decision rule Ŷ = 1{R > t}, with Ŷ = 1 meaning refused (Απόρριψη) and Ŷ = 0 meaning granted (Έγκριση). Y = 1 means δυσκολίες αποπληρωμής. Matches `src/fairness/thresholds.py` and the Ch. 2 guide ¶4.
- (2026-11-21, Claude's choice, user to confirm) No symbol f for the model: the Ch. 2 guide defines X, Y, R, t, Ŷ only, so the decision diagram says "Μοντέλο" and "R ∈ [0, 1]" instead of "Μοντέλο f" and "R = f(X)". If the user adds f to Ch. 2 ¶3, restore it.

## Layout habits

## Assets made

<!-- One line per asset: id: anything later assets must stay consistent with. -->

- score-decision-outcome (2.1): solid = observed, dashed = not observed. Reuse this encoding.
- roc-illustration (2.3): binormal a = 1.19 (AUC ≈ 0,80), threshold point at FPR = 0,40. It is illustrative; the thesis's own ROC is a Ch. 7 result.
