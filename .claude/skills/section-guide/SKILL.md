---
name: section-guide
description: >-
  Guides the user through writing one thesis section themselves: what the
  section must say, its paragraph plan, the evidence and numbers for each
  paragraph, and the references to cite. Also reviews a section the user has
  written against that guide. Use when the user asks for guidance, a plan or an
  outline for a thesis section or chapter they will write, or asks for a review
  of a section they wrote. Learns from every use by updating learnings.md.
argument-hint: "<section, e.g. 5.1 or 'Ch. 6'> | review <section>"
---

# Section guide

The user writes the section; you plan it. **Do not write the thesis prose.** Give the plan, the evidence and the sources so the user can write the Greek text themselves.

Two modes:

- `/section-guide <section>`: **guide mode**. Produce a writing guide for one section (e.g. `5.7`, `6`, `3.2 separation`).
- `/section-guide review <section>`: **review mode**. Check the text the user wrote against the guide and the writing rules, then learn from the differences.

## Step 0: load context (both modes)

Read these every time, in this order:

1. `learnings.md` (next to this file). Learnings **override** the defaults below.
2. `.cursor/rules/thesis-writing.mdc`: the writing rules. Every guide must respect them, especially the citation format `[Full Title]`.
3. Vault `text/` folder (`/Users/c4t/c4t-vault/1-Projects/Uni/thesis/text/`): `structure.md` (outline, RQ per chapter, Phase-step → chapter table), `writing-guide.md` (the story), `writting-rules.md`, and `glossary.md` if it exists.
4. The chapter text already written in `text/` (`.docx` → read with `pandoc <file> -t markdown`), to avoid repeating what earlier sections already say and to match the user's voice.
5. Earlier guides in `text/guides/` for neighbouring sections.

If the section is not in `structure.md`, ask the user where it belongs before going on.

## Step 1: gather the material (guide mode)

Find everything the section can draw on. Read it, don't guess.

- **Writing notes:** `Notes/writing-notes/phase-*/` (use the Phase-step → chapter table in `structure.md`).
- **Decisions:** `docs/decisions.md` rows (`accepted` / `closed`) and `docs/open-questions.md`. Open items become TODOs, not content.
- **Repo docs:** `docs/*.md` (e.g. `feature_engineering.md`, `credit_scoring.md`, `fairness_audit.md`, `EDA.md`).
- **Numbers and figures:** `artifacts/` (e.g. `artifacts/fairness/summary_baseline/`, `artifacts/fairness/<model>/baseline/report.md`, `artifacts/runs/summary_baseline.csv`) and the notebooks in `analysis/notebooks/`. Record the exact file for every number.
- **Sources:** vault `bibliography/*-notes/` and `Notes/audit/`, `Notes/mitigation/`. Use `grep -ril <keyword>` over the vault to find which notes cover the section's topics. Take each citation title from the note's `title` frontmatter (or the official title if there is none). Note where in the source the point is (section, page, equation) when the note says so.

If the section needs a source that is not in the bibliography, say so and suggest what kind of source is missing (don't invent one; you may name well-known candidate works for the user to check and add).

## Step 2: write the guide (guide mode)

Use this template. Write it in English. Use the Greek terms from `glossary.md` where they exist.

```markdown
# Guide: <section number> <section title>

**Serves:** RQ<n> | **Role in the story:** <1–2 sentences, from writing-guide.md>
**Length:** ~<n> paragraphs, ~<n> pages
**Takes from the previous section:** <what the reader already knows>
**Hands to the next section:** <what this section must leave the reader with>

## The point of this section
<2–3 sentences: the one thing the reader should take away.>

## Paragraph plan

### ¶1 <short label>
- **Purpose:** <one sentence: what this paragraph does>
- **Says:** <bullet points of content, in order>
- **Evidence:** <numbers with source file; figure/table with artifact path>
- **Cite:** [Full Title] — <what the citation supports, where in the source>
- **Watch out:** <claim-discipline or framing risk, if any>

### ¶2 ...

## Figures and tables
| # | What | Source file | What the reader should see | Referenced in ¶ |

## References
| Source [Full Title] | Must / optional | Supports | Note in vault |

## Terms
| English | Greek (glossary) | First used in ¶ |
<Mark terms with no glossary entry as NEW.>

## Open items
- [TODO: ...] <missing source, open decision, number not yet produced>

## Self-check before you finish
- [ ] <the 4–6 rules from thesis-writing.mdc most at risk in this section>
```

Rules for the content:

- Each paragraph does **one** job. Open the section with the question it answers; close it with the answer and the handover.
- Every factual claim in the plan has either a number with its source file or a citation. No unsupported claims.
- Numbers follow rule 5 of `thesis-writing.mdc` (group values, signed gap, 95% interval, base rate, threshold).
- Prefer the user's own decisions and wording from the writing notes. Point to the exact note section to draw on.
- Keep the guide to what this section needs. Theory that later chapters don't use goes out.

## Step 3: save and reply (guide mode)

- Save the guide to `text/guides/<section>-<slug>.md` in the vault (e.g. `text/guides/5.7-fairness-metrics.md`). Create the folder if needed. If a guide for that section exists, update it and say what changed.
- In the reply: the link, the section's point in one sentence, the paragraph count, the open items, and anything you need the user to decide.

## Review mode

1. Read the user's text for the section (from `text/`, via pandoc for `.docx`) and its guide in `text/guides/`.
2. Check it against the guide and `thesis-writing.mdc`: citations in `[Full Title]` form and present in the bibliography; numbers match their source files; claim discipline; notation and glossary terms; every figure/table captioned and referenced; the opening question and closing handover.
3. Report findings as a short list, most important first, each with the paragraph and a suggested fix. Do not rewrite the user's paragraphs; quote the problem and say what to change.
4. Note where the user's text **deliberately differs** from the guide (a different paragraph order, a dropped point, an added source). Ask about any difference whose reason isn't obvious. These differences are the main input for learning.

## Step 4: learn (both modes, every time)

At the end of every use, update `learnings.md`:

- Record what the user corrected, rejected, asked for again, or did differently from the guide, and what they confirmed worked. Write it as a **general rule** for future guides, not a note about one paragraph. Example: "The user wants the formula in the paragraph that first uses a metric, not in a separate definitions paragraph."
- Put section-specific facts (e.g. "Ch. 6 already defines base rate in ¶2") under **Section notes**, so later guides don't repeat them.
- Update or replace an existing entry rather than adding a contradicting one. Keep the file short: merge similar entries.
- If a learning changes the guide **template** itself, edit the template in this file and record the change in `learnings.md`.
- If a learning is a writing rule for all thesis text (not just guides), don't put it here: propose adding it to `.cursor/rules/thesis-writing.mdc` and ask the user.
- If a learning contradicts a decision in `docs/decisions.md`, don't record it; point out the conflict to the user.
- Tell the user in one line what you added or changed in `learnings.md`.
