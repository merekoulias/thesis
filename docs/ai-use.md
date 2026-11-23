# AI use log

What generative AI did in the thesis deliverables: the **code** (this repo), the **thesis text** (`text/` in the vault) and the **presentation**. This file is the source for the AI-use statement in the thesis. Check with the supervisor or department whether there is an official declaration template. As of 2026-11-21, `text/writting-rules.md` says nothing about AI.

Keep it current: add a row whenever AI produces or changes something that ends up in a deliverable. Log one row per unit of work (a module or plan phase, a section, a slide), not one row per prompt.

## 1. Tools

| Tool | Model(s) | Used for | Period |
|---|---|---|---|
| Cursor (agent: Composer) | Composer (exact model not recorded) | Code: executing the phases of the implementation plans | 2026-10-21 – 2026-10-23 |
| Claude Code | Claude Opus (version not recorded); Claude Sonnet 5.5 (one commit) | Code: phase reviews and review fixes | 2026-10-21 – 2026-10-23 |
| Claude Code | Claude Opus 5.5 | Infrastructure for frozen thesis assets, the `thesis-asset` skill, this log; thesis figures | 2026-11-21 – |
| Claude Code (`section-guide` skill) | Claude (version not recorded) | Thesis text: section plans and reviews of the user's drafts (does not write the text) | 2026-11-17 – |
| _TODO_ | | | |

## 2. Ground rules

These are the rules that applied to AI output. They are enforced through `CLAUDE.md`, `.cursor/rules/` and `.claude/skills/`.

- The author sets every methodological decision (`docs/decisions.md`). AI does not record decisions the author has not stated.
- AI never invents references. The text cites only sources in the vault's `bibliography/`, and anything missing becomes a `[TODO: source]`.
- Every number in the text comes from `artifacts/`, the notebooks or the writing notes, never from a model's memory (`.cursor/rules/thesis-writing.mdc`). Key numbers are registered in `thesis_assets/numbers.yaml`.
- Sections the author writes are guided, not written, by AI (`section-guide` skill). AI may insert citations and English terms into them only as tracked changes, which the author accepts or rejects (`annotate-paragraph` skill).
- Code produced by AI passes the repo's tests and a separate review before it is accepted. The author reviews all of it and is responsible for all of it.

## 3. Log

**AI role** (what the AI did): `generated` (wrote it from a spec) · `co-written` (AI and author both wrote substantial parts) · `edited` (language, style or formatting changes to the author's work) · `reviewed` (feedback or a verdict only, no changes) · `fixed` (applied changes after a review) · `planned` (outline or plan, no final content) · `research` (summaries of sources, look-ups) · `explained`.
**Extent:** `none` · `minor` · `substantial` · `all`.
**Evidence:** a commit, a plan, a file or a chat. Commits made by an agent carry a `Co-authored-by:` trailer (`git log --format='%h %s %(trailers:key=Co-Authored-By,valueonly)'`).

### 3.1 Code (this repo)

| Date | Location | Tool / model | AI role | Author's role | Extent | Evidence |
|---|---|---|---|---|---|---|
| 2025-10 – 2026-09 | `analysis/`, `src/data/`, `src/features/`, first LightGBM model, EDA notebooks, `.cursor/rules/`, `CLAUDE.md` | _TODO_ | _TODO_ | _TODO_ | _TODO_ | No `Co-authored-by` trailers on these commits. **Author to fill in.** |
| 2026-10-21 | `.cursor/plans/multi-model-nested-cv.md` (implementation plan) | _TODO_ | _TODO_ | _TODO_ | _TODO_ | `03a9797` |
| 2026-10-21 – 22 | `src/models/` (nested-CV credit scorers), its tests and configs: plan phases 0–9 | Cursor Composer | generated | wrote the plan, ran each phase, accepted it after review | substantial | Commits `models: phase N` with the trailer `Co-authored-by: Cursor`; plan header "Executor: Cursor Composer" |
| 2026-10-21 – 22 | `src/models/`: review of each phase and fixes to its findings | Claude Code (Opus) | reviewed, fixed | decided on the verdicts | substantial | Plan progress log ("Review verdict"); commits between phases (`Enhance …`, `Refine …`, `Update phase N status …`) |
| 2026-10-22 | `.cursor/plans/fairness-audit.md` (implementation plan) | _TODO_ | _TODO_ | _TODO_ | _TODO_ | `a9eac5a`, `51e7a26` |
| 2026-10-22 – 23 | `src/fairness/` (baseline fairness audit), its tests, `configs/fairness.yaml`, `docs/fairness_audit.md`: plan phases 0–9 | Cursor Composer | generated | wrote the plan, ran each phase, accepted it after review | substantial | Commits `fairness: phase N` with the trailer `Co-authored-by: Cursor` |
| 2026-10-22 – 23 | `src/fairness/`: review of each phase and fixes to its findings | Claude Code (Opus); Claude Sonnet 5.5 for `725fd17` | reviewed, fixed | decided on the verdicts | substantial | Plan progress log ("fixed by the reviewer"); `725fd17` (trailer `Claude Sonnet 5.5`); commits between phases |
| 2026-11-21 | `thesis_assets/` (build/check script incl. draw.io export, manifest, numbers register, `render/style.py`, README), `tests/test_thesis_assets_build.py`, `.claude/skills/thesis-asset/`, `docs/ai-use.md`, `.gitignore`, `CLAUDE.md` | Claude Code (Opus 5.5) | generated | asked for and specified the structure and the skill's interface; reviews it | all | This session |
| 2026-11-21 | `thesis_assets/render/ch2.py:roc_illustration` (illustrative ROC curve) | Claude Code (Opus 5.5), `thesis-asset` | generated | specified the content, labels and style in detail | substantial | Manifest `roc-illustration` |
| 2026-11-21 | `thesis_assets/diagrams/score-decision-outcome.drawio` (score → decision → outcome diagram) | Claude Code (Opus 5.5), `thesis-asset` | generated | specified every node, label, line style and the layout | substantial | Manifest `score-decision-outcome` |
| 2026-11-23 | `.claude/skills/section-guide/` (`SKILL.md` template: guiding questions with clues per paragraph, sourcing limited to the collected bibliography; paragraph and section review with extra paragraph checks; `learnings.md`), new `.claude/skills/annotate-paragraph/` (citations and English terms as tracked changes), `.cursor/rules/thesis-writing.mdc` rule 1 (page for direct quotes), vault `text/writing-guide.md` (notation X, A, G), `CLAUDE.md` | Claude Code (Opus 5.5) | edited | designed the question-and-clue pattern and its rules; reviews the skill | substantial | This session |

### 3.2 Thesis text (vault `text/`)

| Date | Chapter / section | Tool / model | AI role | Author's role | Extent | Evidence |
|---|---|---|---|---|---|---|
| 2026-11-17 | Ch. 2 (plan for the whole chapter) | Claude Code, `section-guide` | planned | will write the text | none in the text so far | `text/guides/2-credit-scoring-and-ml.md` |
| _TODO_ | `3-fairness-in-ML.docx` | _TODO_ | _TODO_ | _TODO_ | _TODO_ | **Author to fill in.** |
| 2026-11-21 | Ch. 2: captions of figures `score-decision-outcome` (2.1) and `roc-illustration` (2.3), proposed in `thesis_assets/manifest.yaml` | Claude Code (Opus 5.5), `thesis-asset` | generated | to review and adopt or rewrite | caption text only | Manifest |
| 2026-11-21 | Ch. 2 ¶1 (opening, history lead-in): facts and sources for the history, a 3-sentence English summary of it, answers to the chapter question; review of the author's Greek draft; a corrected Greek version of the draft with citations (first rejected by the author, then adopted with the author's changes as the final ¶1); placement of citations in the author's own draft; new sources found and added to the vault (Federal Reserve 2007 report; Durand 1941, Ch. 4; Lessmann et al. manuscript) | Claude Code (Opus 5.5), `section-guide` | research, reviewed, edited | wrote the first Greek draft; chose the AI's corrected version and revised it (split and reworded the 1958 / 1960s–70s sentences, wording changes) | substantial: the final ¶1 is largely the AI's corrected version, including its citations and the roadmap sentence | `text/guides/2-credit-scoring-and-ml.md` ¶1; this session | `text/guides/2-credit-scoring-and-ml.md` ¶1; vault `paper-usage.md` (Fed report entry); this session |
| 2026-11-21 | Ch. 2 ¶2: content questions for the paragraph, review of the author's draft, and the closing sentence (handover to ¶3) written at the author's request (two variants offered) | Claude Code (Opus 5.5), `section-guide` | planned, reviewed, generated (one sentence) | wrote the rest of ¶2; chooses or edits the sentence | one sentence | this session |
| 2026-11-21 | Vault `text/glossary.md` (created; 14 term pairs from the two Ch. 2 figures) | Claude Code (Opus 5.5), `thesis-asset` | co-written | chose the Greek terms in the figure specs | minor | `text/glossary.md` |

### 3.3 Presentation

| Date | Slide(s) | Tool / model | AI role | Author's role | Extent | Evidence |
|---|---|---|---|---|---|---|
| | | | | | | |

### 3.4 Research support (not a deliverable, but it feeds the text)

| Date | What | Tool / model | AI role | Author's role | Evidence |
|---|---|---|---|---|---|
| 2026-10 – | Paper and directive notes in the vault (`bibliography/papers-notes/`, `directive-notes/`): summaries and answers to phase questions, each with a section/page and a quote | Claude Code | research | chose the papers and the questions | Note template `bibliography/papers-notes/_template.md` |
| _TODO_ | | | | | |

## 4. Open items for the author

- Fill in the `_TODO_` rows: code before 2026-10-21, both implementation plans, `3-fairness-in-ML.docx`, and any other tool (ChatGPT, Copilot, translation tools).
- Ask the supervisor or department whether an AI-use declaration is required and in what form.
