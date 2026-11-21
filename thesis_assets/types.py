"""Types for the frozen thesis assets (``thesis_assets/``)."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

ASSET_KINDS: Tuple[str, ...] = ("figure", "table")
"""Allowed values of ``kind``; each kind has its own output folder."""

KIND_DIRS: Dict[str, str] = {"figure": "figures", "table": "tables"}
"""Output folder (relative to ``thesis_assets/``) for each asset kind."""

USED_IN_VALUES: Tuple[str, ...] = ("thesis", "slides")
"""Allowed values of ``used_in``."""


@dataclass(frozen=True)
class AssetSpec:
    """One figure or table as declared in ``manifest.yaml``.

    Attributes:
        id: Stable slug; also the file-name stem of every output file.
        kind: ``"figure"`` or ``"table"``.
        chapter: Section where the asset currently appears (e.g. ``"6.2"``).
        caption: Caption exactly as in the thesis (Greek).
        used_in: Deliverables that use the asset (``thesis``, ``slides``).
        copy: Repo-relative file to copy as-is (mutually exclusive with
            ``render``).
        render: ``"module:function"`` that writes the asset (mutually exclusive
            with ``copy``).
        inputs: Repo-relative files the asset is computed from; their hashes are
            recorded so ``--check`` can tell when the asset is stale.
        producer: Command that produced the inputs (for the reader of the
            manifest; not executed).
        notes: Free text.
    """

    id: str
    kind: str
    chapter: str
    caption: str
    used_in: Tuple[str, ...]
    copy: Optional[str] = None
    render: Optional[str] = None
    inputs: Tuple[str, ...] = ()
    producer: str = ""
    notes: str = ""


@dataclass(frozen=True)
class NumberSpec:
    """One number quoted in the thesis text, as declared in ``numbers.yaml``.

    Attributes:
        id: Stable slug.
        value: The value exactly as written in the text (already rounded).
        decimals: Number of decimals used in the text.
        chapter: Section where the number appears.
        used_in: Deliverables that quote the number.
        path: Repo-relative JSON or CSV file holding the number.
        key: For JSON: dotted key (list indices allowed, e.g. ``folds.0.auc``).
        where: For CSV: column → value filter that selects exactly one row.
        column: For CSV: column holding the number.
        notes: Free text.
    """

    id: str
    value: float
    decimals: int
    chapter: str
    used_in: Tuple[str, ...]
    path: str
    key: Optional[str] = None
    where: Dict[str, str] = field(default_factory=dict)
    column: Optional[str] = None
    notes: str = ""


@dataclass
class BuildRecord:
    """Provenance of one built asset, stored in ``provenance.json``.

    Attributes:
        built_at: ISO-8601 timestamp (UTC) of the build.
        git_commit: ``HEAD`` commit of the repo at build time.
        git_dirty: Whether the working tree had uncommitted changes.
        inputs: Repo-relative input path → SHA-256.
        outputs: Repo-relative output path → SHA-256.
    """

    built_at: str
    git_commit: str
    git_dirty: bool
    inputs: Dict[str, str]
    outputs: Dict[str, str]


@dataclass
class CheckReport:
    """Result of ``--check``.

    Attributes:
        errors: Problems that make the frozen assets untrustworthy.
        warnings: Things worth knowing that are not errors (e.g. inputs absent
            on this machine because ``artifacts/`` is gitignored).
    """

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
