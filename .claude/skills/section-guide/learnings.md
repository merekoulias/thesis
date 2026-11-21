# Section-guide learnings

What the section-guide skill has learned from the user. Read before every use; entries here override the defaults in `SKILL.md`. Keep entries short, general and dated. Merge or replace entries rather than adding contradicting ones.

## Guide format

<!-- How the user wants guides laid out: length, level of detail, order of parts. -->

## Content and sources

<!-- What to include or leave out, how to pick and present citations, evidence and numbers. -->

- (2026-11-17) Sourcing facts: the vault book notes number the chapters differently from the files ("6. Testing" = `6-legal.pdf`, "7. Broader view" = `7-testing.pdf`). Get page numbers with `pdftotext -layout <chapter>.pdf - | awk 'BEGIN{p=1} /\f/{p++} {print p": "$0}'`. The counter matches the printed page for the book chapters, Kozodoi, Hurlin and the Wachter SSRN preprint.
- (2026-11-17) Model-method source: the user chose to cite [The Elements of Statistical Learning: Data Mining, Inference, and Prediction] for LR (§4.4, §18.3.2), trees/bagging/RF (§9.2, §8.7, Ch. 15), gradient boosting (§10.9–10.12), MLP (§11.3–11.5) and cross-validation (§7.10). Pages are from the 12th printing. Add a primary paper only when the text names something the book lacks (LightGBM internals → Ke et al. 2017; ReLU/Adam → a deep-learning source).
- (2026-11-17) The AI Act (`bibliography/Directive/Regulation-EU-2024-1689.pdf`) and ESL (`bibliography/books/…Elements-of-Statistical-Learning….pdf`) are in the vault, without notes. Cite them by their official full titles. AI Act provisions checked in the PDF: Recital 58, Art. 6(2)–(3), Art. 10(2)(f)–(g), Annex III 5(b). Art. 10(5) concerns GDPR special categories, which do not include sex or age, so don't use it. There is still no paper note for Wachter et al.
- (2026-11-17) The user delegates bookkeeping for new sources: when a source is added to the vault, write its entry in `bibliography/general-notes/<type>-usage.md` following that file's conventions, with pages checked in the PDF.
- (2026-11-17) Legal depth (user decision): law is a frame in Ch. 2 §2.4 only. No separate legal chapter, no compliance claims, GDPR Art. 22 / SCHUFA left out.
- (2026-11-17) Notation placement (user decision): Ch. 2 defines X, Y, R, t, Ŷ (and the base rate and TPR/FPR/FNR) as formulas; Ch. 3 adds A, G; 5.2 applies them to Home Credit without redefining them.
- (2026-11-17) Notation, delegated to Claude by the user: $A$ = protected attribute ($A_{\text{sex}}$, $A_{\text{age}}$), $G$ = grouping variable with values $g$, no $S$. Rule 7 of `thesis-writing.mdc` was updated to match.
- (2026-11-17) `text/glossary.md` does not exist yet. Mark proposed Greek terms NEW, and avoid «χαρακτηριστικά» for features, because it collides with «προστατευόμενο χαρακτηριστικό».

## The user's writing style

<!-- Patterns seen in the user's own text (review mode) that guides should match. -->

## Section notes

<!-- Facts about sections already written that later guides must respect, e.g. where a term or number is first defined. Format: `<section>: <fact>`. -->

- Ch. 2 (guide only, not yet written, 2026-11-17): plans to define $X$, $Y$ (1 = payment difficulties), $R$, $t$, $\hat{Y}$ (1 = refused), base rate, TPR / FPR / FNR as performance rates, ROC AUC, the four model families, Directive Art. 6 + Charter Art. 21, and direct vs indirect discrimination. Ch. 3 guides should build on these, not redefine them. Confirm once Ch. 2 is written.
- 5.3: answers the reading of Directive Art. 6 on protected attributes as model inputs (placement delegated by the user, 2026-11-17; the reading itself is still the user's). Ch. 2 only points there.
- Ch. 1 and Ch. 3: no text yet (`3-fairness-in-ML.docx` is empty; `thesis.docx` is the department template).

## Template changes

<!-- Dated log of changes made to the guide template in SKILL.md. -->
