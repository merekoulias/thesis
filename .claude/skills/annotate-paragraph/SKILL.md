---
name: annotate-paragraph
description: >-
  Inserts citations ([Full Title]) and English terms in parentheses into one
  thesis paragraph the user wrote in Greek, as tracked changes in the chapter's
  .docx, without changing any of the user's words. Adds new term pairs to the
  glossary. Use after the user writes a paragraph from a section-guide guide and
  before its section-guide review, or whenever the user asks to add citations,
  references or English terms to their text.
argument-hint: "<section> ¶<n>"
---

# Annotate paragraph

The user wrote the paragraph; you only **insert** citations and English terms. Never change, move or delete a word of theirs. Workflow: `section-guide` (guide) → user writes → **annotate-paragraph** → `section-guide review`. Write everything outside the thesis text in **English**.

## Step 0: load context

1. `learnings.md` (next to this file). Learnings override the defaults below.
2. `.cursor/rules/thesis-writing.mdc`, rules 1 (citations) and 7 (terms).
3. The guide for the section in vault `text/guides/` (`/Users/c4t/c4t-vault/1-Projects/Uni/thesis/text/guides/`): the paragraph's questions and clues.
4. `text/glossary.md`.
5. The chapter `.docx` in `text/` (read with `pandoc <file> -t markdown`): the paragraph, and every earlier paragraph of the chapter.

If a Word lock file `text/~$<name>.docx` exists, the document is open in Word: ask the user to close it before editing.

## Step 1: citations

- Match each sentence to the guide question it answers. The question's clue gives the source: insert `[Full Title]` right after the claim it supports, before the sentence's full stop. Several sources: `[Title A], [Title B]`.
- **Direct quote** (text in quotation marks): add the page, `[Full Title, p. <printed page>]`.
- A sentence that answers no question: search the vault `bibliography/` (notes and PDFs) for a source and check it in the PDF. If none supports it, insert `[TODO: source for <claim>]`. Don't search outside the vault.
- A sentence that needs no citation (a transition, the section's own question, a handover, the thesis's own results with a figure or table reference) gets none.
- Take each title exactly from the note's `title` frontmatter (or the official title).

## Step 2: English terms

- At a technical term's **first use in the chapter** (check the earlier paragraphs), add the English in parentheses after the Greek: «κατώφλι (threshold)». Later uses get nothing.
- Acronyms at first use: Greek term, then the English term and the acronym: «ποσοστό ψευδώς θετικών (false positive rate, FPR)». Later uses may use the acronym alone.
- Use the glossary's English for terms in the glossary. If the user's Greek differs from the glossary's term, don't change it: flag it in the reply.
- **New terms:** if the term is not in the glossary, add the row to `text/glossary.md` automatically (English with acronym, the user's Greek term, first use as `Ch. <n> ¶<n>`, note), keeping the table sorted by English. List the new rows in the reply.
- Everyday words and symbols already defined (X, A, G, Y, R, Ŷ, t) get no English.

## Step 3: apply as tracked changes

- Edit the chapter `.docx` with **tracked changes** (insertions only, author `Claude`), using the `anthropic-skills:docx` skill's tracked-changes workflow. The user accepts or rejects each one in Word.
- Afterwards, compare the paragraph's text before and after with the insertions removed: it must be identical. If not, undo and report.

## Step 4: reply and bookkeeping

- In the reply, a table: `# | inserted | after (quoted words) | supports / reason | source location (path, printed page, PDF page)`. Then the TODOs, the new glossary rows and any term that differs from the glossary.
- If the guide shows the paragraph as `reviewed`, change its status to `changed since review <date>` (and the section status to `outdated` if it was `reviewed`).
- Add or extend the session's row in `docs/ai-use.md` §3.2 (AI role: edited, citations and terms inserted).
- Record in `learnings.md` anything the user rejected or corrected, as a general rule. Tell the user in one line what changed there.
