# Section-guide learnings

What the section-guide skill has learned from the user. Read before every use; entries here override the defaults in `SKILL.md`. Keep entries short, general and dated. Merge or replace entries rather than adding contradicting ones.

## Guide format

<!-- How the user wants guides laid out: length, level of detail, order of parts. -->

- (2026-11-21) The user writes paragraph by paragraph and asks for the *information* behind each one (facts, sources, answers), never wording. Chapter openings may get a short context lead-in (e.g. history) from the user, so plan where its sources come from.
- (2026-11-23) Long vault paths: define an alias once in the guide header (e.g. `FML/` for the fairmlbook chapter folder) and use it in the clues. Give one page when the printed page equals the PDF page, and state the offset in the header for sources where it doesn't.

## Content and sources

<!-- What to include or leave out, how to pick and present citations, evidence and numbers. -->

- (2026-11-17) Sourcing facts: the vault book notes number the chapters differently from the files ("6. Testing" = `6-legal.pdf`, "7. Broader view" = `7-testing.pdf`). Get page numbers with `pdftotext -layout <chapter>.pdf - | awk 'BEGIN{p=1} /\f/{p++} {print p": "$0}'`. The counter matches the printed page for the book chapters, Kozodoi, Hurlin and the Wachter SSRN preprint.
- (2026-11-17) Model-method source: the user chose to cite [The Elements of Statistical Learning: Data Mining, Inference, and Prediction] for LR (§4.4, §18.3.2), trees/bagging/RF (§9.2, §8.7, Ch. 15), gradient boosting (§10.9–10.12), MLP (§11.3–11.5) and cross-validation (§7.10). Pages are from the 12th printing; PDF page = printed + 19. Add a primary paper only when the text names something the book lacks (LightGBM internals → Ke et al. 2017; ReLU/Adam → a deep-learning source).
- (2026-11-23) Lessmann et al., *Benchmarking … A ten-year update* (manuscript, in the vault, printed = PDF page) is the credit-specific companion to ESL: LR with L2 (§3.1, p. 14), ANN (p. 15), bagging, RF and boosting (§3.2, p. 18–19), confusion matrix and threshold (§4.3, p. 29), ROC/AUC and Brier score (p. 29–30), calibration and BS vs AUC (p. 9, 39, 42), selective labels (p. 6). Use it where ESL is not credit-specific or lacks the point (e.g. Brier score). Federal Reserve report: PDF page = printed + 40.
- (2026-11-17) The AI Act (`bibliography/Directive/Regulation-EU-2024-1689.pdf`) and ESL (`bibliography/books/…Elements-of-Statistical-Learning….pdf`) are in the vault, without notes. Cite them by their official full titles. AI Act provisions checked in the PDF: Recital 58, Art. 6(2)–(3), Art. 10(2)(f)–(g), Annex III 5(b). Art. 10(5) concerns GDPR special categories, which do not include sex or age, so don't use it. There is still no paper note for Wachter et al.
- (2026-11-17) The user delegates bookkeeping for new sources: when a source is added to the vault, write its entry in `bibliography/general-notes/<type>-usage.md` following that file's conventions, with pages checked in the PDF.
- (2026-11-17) Legal depth (user decision): law is a frame in Ch. 2 §2.4 only. No separate legal chapter, no compliance claims, GDPR Art. 22 / SCHUFA left out.
- (2026-11-17) Notation placement (user decision): Ch. 2 defines X, Y, R, t, Ŷ (and the base rate and TPR/FPR/FNR) as formulas; Ch. 3 adds A, G; 5.2 applies them to Home Credit without redefining them.
- (2026-11-17) Notation, delegated to Claude by the user: $A$ = protected attribute ($A_{\text{sex}}$, $A_{\text{age}}$), $G$ = grouping variable with values $g$, no $S$. Rule 7 of `thesis-writing.mdc` was updated to match.
- (2026-11-21) `text/glossary.md` now exists: read it before proposing Greek terms, and mark terms not in it NEW. Avoid «χαρακτηριστικά» for features, because it collides with «προστατευόμενο χαρακτηριστικό».
- (2026-11-23) Register a guide's key numbers in `thesis_assets/numbers.yaml` when the guide plans them (the user asked for this before writing ¶12), and name the register ids in the paragraph's Evidence. Percentages use the source flags `percent` / `complement`.
- (2026-11-21) Length planning: the department template (`text/thesis.docx`) is Times New Roman, body 11 pt, 1.5 line spacing, A4 with 3 cm margins top, bottom and left and 2 cm on the right. That gives about 33 lines and 400–450 words per full page, about 13–14 words per line. Give section lengths in lines as well as pages.

## The user's writing style

<!-- Patterns seen in the user's own text (review mode) that guides should match. -->

- (2026-11-21, Ch. 2 ¶2 draft) The user writes in a narrative voice, with rhetorical turns («Κάπου εδώ, η λύση φαίνεται απλή…», «Τουλάχιστον έτσι ήταν αρχικά πιστευτό»). Watch for informal register, and for story arcs ("believed, until it was revealed") that the sources don't support. Recurring language issues: article and case agreement, and «χαρακτηριστικά» used for features (the glossary says «μεταβλητές εισόδου»). Translations of English source phrases can shift meaning ("less money left on the table" became "fewer losses"), so check them against the source.

## Section notes

<!-- Facts about sections already written that later guides must respect, e.g. where a term or number is first defined. Format: `<section>: <fact>`. -->

- Ch. 2 (guide only, not yet written, 2026-11-17): plans to define $X$, $Y$ (1 = payment difficulties), $R$, $t$, $\hat{Y}$ (1 = refused), base rate, TPR / FPR / FNR as performance rates, ROC AUC, the four model families, Directive Art. 6 + Charter Art. 21, and direct vs indirect discrimination. Ch. 3 guides should build on these, not redefine them. Confirm once Ch. 2 is written.
- Ch. 2 ¶1 (2026-11-21): written by the user, with a history lead-in **with dates**. Thomas et al. could not be accessed (Google Books blocks automated search), so the dates are cited from the free Federal Reserve 2007 report on credit scoring, now in `bibliography/papers/` with a `paper-usage.md` entry. ML step: Hurlin p. 6–7, Kozodoi p. 1 and p. 12. ¶1 already says that Regulation B under ECOA recognises statistical scoring, and that logistic regression is the typical scorecard model; later paragraphs shouldn't repeat this.
- (2026-11-21) When the user asks for a source outside the bibliography and the wanted one is paywalled, look for a free primary source with the same facts (government reports, author preprints) before narrowing the claim. Check pages in the PDF itself.
- (2026-11-23) Inserting citations and English terms into the user's text moved to the `annotate-paragraph` skill (with the 2026-11-21 learning about not rewriting the user's draft). Review only checks them.
- (2026-11-21, narrowed 2026-11-23) Guides only plan points the collected sources support; going beyond the bibliography is the user's call alone. In review mode, when the user's own text makes a claim no vault source supports, set out both options (narrow the claim, or add a named source) and let the user choose.
- 5.3: answers the reading of Directive Art. 6 on protected attributes as model inputs (placement delegated by the user, 2026-11-17; the reading itself is still the user's). Ch. 2 only points there.
- (2026-11-23) The Ch. 2 text is in `text/3-fairness-in-ML.docx` (misnamed; it holds Ch. 2 ¶1–¶2). Ch. 1 and Ch. 3 have no text yet; `thesis.docx` is the department template.
- Ch. 2 ¶2 (drafted 2026-11-21, no citations yet): already says that credit scores estimate default risk and guide the decision, the benefits for lenders and borrowers, the systematic-disadvantage concern for groups (gender, age) in access and cost, and that dropping the protected variables fails because of proxies (triangulation). ¶6 and ¶18 refer back instead of repeating it.

## Template changes

<!-- Dated log of changes made to the guide template in SKILL.md. -->

- (2026-11-23, user request) Each paragraph now has Purpose, a 1–2 sentence Says summary, Evidence (exact values) and ordered **guiding questions with clues**; the per-paragraph Cite and Watch out are gone (scope notes move into the clues). Only questions the collected sources answer. Everything is written in English, so NEW terms get no proposed Greek. Clues are specific because the user has not read the whole bibliography and had a three-week break from it. Each clue gives the full title, the PDF's vault path, and printed and PDF page. This replaces the 2026-11-21 "guiding questions" learning. The Ch. 2 guide was moved to this template on 2026-11-23 (old version in `text/guides/archive/`).
- (2026-11-23, user request) Review has two scopes. A paragraph review (`review <section> ¶n`) gives a question-coverage table, checks claims against the clue's source, the writing rules and the Greek, then adds a status line in the guide and a Section note. A section review (`review <section>`) checks what spans paragraphs. Paragraph review also checks wording too close to the source, paragraph shape, first use of terms and acronyms, term consistency with earlier paragraphs, and (Ch. 5–9) `numbers.yaml`, results vs interpretation and pre-empting mitigation. Paragraph reviews never make a section reviewed: the guide's `Section status` tracks that separately, and a later paragraph review marks it outdated.
