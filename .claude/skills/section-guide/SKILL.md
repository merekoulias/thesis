---
name: section-guide
description: >-
  Guides the user through writing one thesis section themselves: what the
  section must say, its paragraph plan with the evidence and exact numbers for
  each paragraph, and ordered guiding questions whose answers form each
  paragraph, with specific clues and source locations. Also reviews a section
  the user has written against that guide. Use when the user asks for guidance,
  a plan, an outline or guiding questions for a thesis section or chapter they
  will write, or asks for a review of a section they wrote. Learns from every
  use by updating learnings.md.
argument-hint: "<section, e.g. 5.1 or 'Ch. 6'> | review <section> [¶n]"
---

# Section guide

The user writes the section; you plan it. **Do not write the thesis prose.** Give the plan, the evidence and guiding questions with clues so the user can write the Greek text themselves. Write everything in **English**.

Workflow per paragraph: guide → the user writes → `annotate-paragraph` (citations and English terms, as tracked changes) → paragraph review. Section review comes once all paragraphs are done.

Two modes:

- `/section-guide <section>`: **guide mode**. Produce a writing guide for one section (e.g. `5.7`, `6`, `3.2 separation`).
- `/section-guide review <section> ¶<n>`: **paragraph review**. Check one paragraph the user wrote against its guiding questions, its sources and the writing rules.
- `/section-guide review <section>`: **section review**. Check the section as a whole. Paragraph reviews never count as a section review.

## Step 0: load context (all modes)

Read these every time, in this order:

1. `learnings.md` (next to this file). Learnings **override** the defaults below.
2. `.cursor/rules/thesis-writing.mdc`: the writing rules. The user's text must respect them, especially the citation format `[Full Title]`.
3. Vault `text/` folder (`/Users/c4t/c4t-vault/1-Projects/Uni/thesis/text/`): `structure.md` (outline, RQ per chapter, Phase-step → chapter table), `writing-guide.md` (the story), `writting-rules.md` (department formatting) and `glossary.md`.
4. The chapter text already written in `text/` (`.docx` → read with `pandoc <file> -t markdown`), to avoid repeating what earlier paragraphs already say.
5. Earlier guides in `text/guides/` for neighbouring sections.

If the section is not in `structure.md`, ask the user where it belongs before going on.

## Step 1: gather the material (guide mode)

Find everything the section can draw on. Read it, don't guess.

- **Writing notes:** `Notes/writing-notes/phase-*/` (use the Phase-step → chapter table in `structure.md`). If the phase the section depends on is not finished, say so and stop.
- **Decisions:** `docs/decisions.md` rows (`accepted` / `closed`) and `docs/open-questions.md`. Open items become TODOs, not content.
- **Repo docs:** `docs/*.md` (e.g. `feature_engineering.md`, `credit_scoring.md`, `fairness_audit.md`, `EDA.md`).
- **Numbers and figures:** `artifacts/` (e.g. `artifacts/fairness/summary_baseline/`, `artifacts/runs/summary_baseline.csv`) and `analysis/notebooks/`. Record the exact value and file for every number.
- **Sources:** the PDFs in vault `bibliography/`, their notes in `bibliography/*-notes/`, `Notes/audit/` and `Notes/mitigation/`. Use `grep -ril <keyword>` over the vault to find which notes cover the section's topics. Take each title from the note's `title` frontmatter (or the official title if there is none). **Verify every page in the PDF itself** (command in `learnings.md`), not from the notes.

**Only the collected sources count.** A point that no vault source (or, for the user's own work, no writing note, decision, doc or artifact) supports gets no question and leaves the plan. List it under Open items. Going beyond the bibliography is the user's decision alone (as for the history lead-in of Ch. 2 ¶1); search for outside sources only when the user asks.

## Step 2: write the guide (guide mode)

Use this template.

```markdown
# Guide: <section number> <section title>

**Serves:** RQ<n> | **Role in the story:** <1–2 sentences, from writing-guide.md>
**Length:** ~<n> paragraphs, ~<n> lines, ~<n> pages
**Section status:** not reviewed
**Takes from the previous section:** <what the reader already knows>
**Hands to the next section:** <what this section must leave the reader with>

## The point of this section
<2–3 sentences: the one thing the reader should take away.>

## Paragraph plan

### ¶1 <short label>
- **Purpose:** <one sentence: what this paragraph does>
- **Says:** <1–2 sentence summary of the content>
- **Evidence:** <exact numbers with source file; figure/table with artifact path, or "—">
- **Questions:**
  1. <content question>
     - *Clue:* <specific facts the answer needs> — [Full Title], `<vault path to the PDF>`, p. <printed> (PDF p. <n>), <§ / Art. / Eq.>
  2. ...

### ¶2 ...

## Figures and tables
| # | What | Source file | What the reader should see | Referenced in ¶ |

## References
| Source [Full Title] | File in vault | Used in ¶ |

## Terms
| English | Greek (from glossary.md, else NEW) | First used in ¶ |

## Open items
- [TODO: ...] <open decision, number not yet produced, point dropped for lack of a source>
```

### Guiding questions

- **The answers, in order, are the paragraph.** Every answer goes into the paragraph; the questions are not a menu to choose from. One question per sentence or two, so the line budget sets the count (a 10–14-line paragraph ≈ 4–7 questions).
- **Content questions only** ("What does a credit score estimate?"), never meta-questions about the paragraph's job, framing or claim discipline.
- **Opening, transition and handover sentences** get content questions too ("What does the reader still not know that ¶4 answers?"); their clue names the paragraph or section they connect to and what it hands over.
- A paragraph that introduces a figure or table has a question about what it shows.
- No question repeats what an earlier paragraph or section already says.

### Clues

The user has not read every source in the bibliography and may not remember recent reading, so clues must be **specific enough to answer without rereading the whole source**, but never the finished sentence.

- **Give:** the key facts the answer needs (names, dates, terms, the defining idea), the source's own term in a quote of at most a few words, and a scope note where the source is narrower than an easy answer ("source says *can*, not *does*").
- **Don't give:** a full sentence, a paraphrase that only needs translating, or a long quote.
- **Location, for a bibliography source:** the full title (also the citation), the PDF's vault path, the printed page and the PDF page when they differ, and the section, article, recital or equation. For a source without pages (HTML), give the section heading.
- **Location, for the user's own work:** the exact value with its artifact path (and column or row), the writing note's path and heading, or the `decisions.md` row.

## Step 3: save and reply (guide mode)

- Save the guide to `text/guides/<section>-<slug>.md` in the vault (e.g. `text/guides/5.7-fairness-metrics.md`). If a guide for that section exists, update it and say what changed.
- **Updating an existing guide:** the vault has no version history, so first copy the old guide to `text/guides/archive/<file>-<date>.md`. Paragraphs the user has already written keep their plan and status and get no questions, unless the user asks. Carry over still-valid decisions recorded in the old guide (e.g. points dropped for lack of a source).
- In the reply: the link, the section's point in one sentence, the paragraph count, the open items, and anything the user must decide.

## Review mode

Two scopes. A section is reviewed only after a **section review**, even if every paragraph has had a paragraph review.

### Paragraph review: `review <section> ¶<n>`

1. **Get the text.** Use the paragraph the user pasted; otherwise find it in the chapter `.docx` in `text/` (via pandoc). Read its plan and questions in the guide, and the paragraphs before and after it.
2. **Question coverage.** Match each sentence to the question it answers. Report it as a table: `Q# | answered? | the sentence (quoted, shortened)`. Then list unanswered questions, answers out of order, and sentences that answer no question (additions: check that a collected source supports them).
3. **Content against the sources.** For each claim, open the source at the clue's location and check that the claim says what the source says: scope (*can* vs *does*), dates, numbers (exact match with Evidence), and English phrases whose meaning shifted in translation. Flag any sentence that is a near-translation of a source sentence without quotation marks; a direct quote needs quotation marks and `[Full Title, p. <n>]`.
4. **Paragraph shape.** It does the one job in its **Purpose**; the first sentence says what the paragraph is about; the sentences connect into an argument rather than reading as a list of answers; it fits its line budget.
5. **Writing rules.** Check against `thesis-writing.mdc`:
   - Citations (inserted by `annotate-paragraph`, which runs before review): `[Full Title]` right after the claim it supports, in the bibliography. If they are missing, point the user to `annotate-paragraph` instead of adding them.
   - Terms: at first use in the chapter, the Greek term with the English in parentheses, and acronyms spelled out (rule 7); the same Greek term as earlier paragraphs and the glossary.
   - Notation; figure/table references; no repetition of earlier paragraphs; the handover to the next paragraph.
   - Ch. 5–9: every key number is registered in `thesis_assets/numbers.yaml`; results stay apart from interpretation (rule 9); nothing pre-empts the mitigation chapters (rule 11).
   - The Greek: grammar (article and case agreement) and register.
6. **Report** in English, findings most important first, each as: quoted problem → what is wrong → what to change. Do not rewrite the paragraph. For a claim with no collected source, set out both options (narrow it, or add a named source) and let the user choose.
7. **Differences from the guide.** Note where the text **deliberately differs** (a different order, a dropped question, an added source) and ask about any difference whose reason isn't obvious. These differences are the main input for learning.
8. **Bookkeeping.** In the guide, add `- **Status:** reviewed <date>` under the paragraph's heading. If the section status says `reviewed`, change it to `outdated (¶<n> reviewed again <date>)`. Under **Section notes** in `learnings.md`, record what the paragraph now says that later paragraphs must not repeat.

### Section review: `review <section>`

1. **Get the text.** Read the whole section, its guide, the end of the previous section and the start of the next.
2. **Paragraph status.** List the paragraphs with no paragraph review, or whose text changed since it. Don't repeat the paragraph checks unless the user asks.
3. **Section checks.** These are things no paragraph review can see:
   - The section opens with its question and closes with the answer and the handover (rule 3), and matches the guide's "Takes from" and "Hands to".
   - Every planned paragraph is there, in a logical order, and each leads into the next.
   - Nothing is said twice across paragraphs or sections, and nothing the section needs is missing.
   - One Greek term and one notation per concept across all paragraphs.
   - Every figure and table is captioned, referenced, and introduced before or where it is discussed.
   - The section serves its RQ, and its length fits the budget.
4. **Report and differences** as in paragraph review steps 6–7.
5. **Bookkeeping.** Set the guide's `**Section status:**` to `reviewed <date>`.

## Step 4: learn (all modes, every time)

At the end of every use, update `learnings.md`:

- Record what the user corrected, rejected, asked for again, or did differently from the guide, and what they confirmed worked. Write it as a **general rule** for future guides, not a note about one paragraph.
- Put section-specific facts (e.g. "Ch. 6 already defines base rate in ¶2") under **Section notes**, so later guides don't repeat them.
- Update or replace an existing entry rather than adding a contradicting one. Keep the file short: merge similar entries.
- If a learning changes the guide **template** itself, edit the template in this file and record the change in `learnings.md`.
- If a learning is a writing rule for all thesis text (not just guides), don't put it here: propose adding it to `.cursor/rules/thesis-writing.mdc` and ask the user.
- If a learning contradicts a decision in `docs/decisions.md`, don't record it; point out the conflict to the user.
- Tell the user in one line what you added or changed in `learnings.md`.
- Add or extend the session's row in `docs/ai-use.md` §3.2.
