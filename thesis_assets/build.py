"""Build, freeze and check the tables and figures used in the thesis.

Every asset in ``thesis_assets/manifest.yaml`` is copied from a pipeline
artifact, rendered by a ``module:function``, or exported from a ``.drawio`` file
with the draw.io desktop CLI. The output is written to
``thesis_assets/figures/`` or ``thesis_assets/tables/`` and its provenance (git
commit, input and output hashes) to ``thesis_assets/provenance.json``.

Usage (from the repo root)::

    python -m thesis_assets.build                 # build every asset
    python -m thesis_assets.build --only ID ...   # build some assets
    python -m thesis_assets.build --check         # validate, build nothing

``--check`` exits with status 1 when an asset is missing, was edited by hand,
is stale (an input changed since the build), when a file has no manifest entry,
or when a number in ``numbers.yaml`` no longer matches its source.
"""

import argparse
import csv
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import yaml

from thesis_assets.types import (
    ASSET_KINDS,
    DIAGRAM_SOURCE_DIR,
    KIND_DIRS,
    USED_IN_VALUES,
    AssetSpec,
    BuildRecord,
    CheckReport,
    NumberSpec,
)

ASSETS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ASSETS_DIR.parent
MANIFEST_FILE = "manifest.yaml"
NUMBERS_FILE = "numbers.yaml"
PROVENANCE_FILE = "provenance.json"

_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_ASSET_KEYS: Set[str] = {
    "id",
    "kind",
    "chapter",
    "caption",
    "used_in",
    "source",
    "notes",
}
_SOURCE_KEYS: Set[str] = {"copy", "render", "drawio", "inputs", "producer"}
_SOURCE_MODES: Tuple[str, ...] = ("copy", "render", "drawio")
_DRAWIO_CANDIDATES: Tuple[str, ...] = (
    "drawio",
    "draw.io",
    "/Applications/draw.io.app/Contents/MacOS/draw.io",
)
_PNG_DPI = 300
_SVG_DPI = 96  # draw.io and Word both treat 1 px as 1/96 in
_NUMBER_KEYS: Set[str] = {
    "id",
    "value",
    "decimals",
    "chapter",
    "used_in",
    "source",
    "notes",
}
_NUMBER_SOURCE_KEYS: Set[str] = {"path", "key", "where", "column"}


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #


def _read_yaml_list(path: Path, top_key: str) -> List[Dict[str, Any]]:
    """Read ``path`` and return the list under ``top_key`` (empty if absent)."""
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or set(raw) - {top_key}:
        raise ValueError(f"{path}: expected a mapping with the single key {top_key!r}")
    entries = raw.get(top_key) or []
    if not isinstance(entries, list):
        raise ValueError(f"{path}: {top_key!r} must be a list")
    return entries


def _check_common(
    entry: Dict[str, Any], allowed: Set[str], where: str, seen: Set[str]
) -> Tuple[str, Tuple[str, ...]]:
    """Validate the fields shared by assets and numbers; return id and used_in."""
    unknown = set(entry) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    asset_id = entry.get("id")
    if not isinstance(asset_id, str) or not _ID_PATTERN.match(asset_id):
        raise ValueError(f"{where}: 'id' must be a lowercase-hyphen slug")
    if asset_id in seen:
        raise ValueError(f"{where}: duplicate id {asset_id!r}")
    seen.add(asset_id)
    chapter = entry.get("chapter")
    if not isinstance(chapter, (str, int, float)) or str(chapter) == "":
        raise ValueError(f"{where} ({asset_id}): 'chapter' is required")
    used_in = entry.get("used_in")
    if (
        not isinstance(used_in, list)
        or not used_in
        or set(used_in) - set(USED_IN_VALUES)
    ):
        raise ValueError(
            f"{where} ({asset_id}): 'used_in' must be a non-empty list of "
            f"{list(USED_IN_VALUES)}"
        )
    return asset_id, tuple(used_in)


def load_manifest(path: Path) -> List[AssetSpec]:
    """Load and validate the asset manifest.

    Args:
        path: Path to ``manifest.yaml``.

    Returns:
        One ``AssetSpec`` per entry, in file order.

    Raises:
        ValueError: If an entry is malformed, an id repeats, or the source does
            not name exactly one of ``copy`` / ``render`` / ``drawio``.
    """
    specs: List[AssetSpec] = []
    seen: Set[str] = set()
    for i, entry in enumerate(_read_yaml_list(path, "assets")):
        where = f"{path.name} entry {i}"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: must be a mapping")
        asset_id, used_in = _check_common(entry, _ASSET_KEYS, where, seen)
        if entry.get("kind") not in ASSET_KINDS:
            raise ValueError(
                f"{where} ({asset_id}): 'kind' must be one of {ASSET_KINDS}"
            )
        caption = entry.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"{where} ({asset_id}): 'caption' is required")
        source = entry.get("source")
        if not isinstance(source, dict) or set(source) - _SOURCE_KEYS:
            raise ValueError(
                f"{where} ({asset_id}): 'source' must be a mapping with keys "
                f"from {sorted(_SOURCE_KEYS)}"
            )
        modes = [m for m in _SOURCE_MODES if source.get(m) is not None]
        if len(modes) != 1:
            raise ValueError(
                f"{where} ({asset_id}): source needs exactly one of {_SOURCE_MODES}"
            )
        copy, render, drawio = (source.get(m) for m in _SOURCE_MODES)
        if (entry["kind"] == "diagram") != (drawio is not None):
            raise ValueError(
                f"{where} ({asset_id}): 'drawio' is the source of every diagram "
                f"and only of diagrams"
            )
        if drawio is not None and not str(drawio).endswith(".drawio"):
            raise ValueError(f"{where} ({asset_id}): 'drawio' must be a .drawio file")
        if render is not None and (not isinstance(render, str) or ":" not in render):
            raise ValueError(
                f"{where} ({asset_id}): 'render' must be 'module:function'"
            )
        inputs = source.get("inputs") or []
        if not isinstance(inputs, list) or not all(isinstance(p, str) for p in inputs):
            raise ValueError(f"{where} ({asset_id}): 'inputs' must be a list of paths")
        specs.append(
            AssetSpec(
                id=asset_id,
                kind=entry["kind"],
                chapter=str(entry["chapter"]),
                caption=caption.strip(),
                used_in=used_in,
                copy=copy,
                render=render,
                drawio=drawio,
                inputs=tuple(inputs),
                producer=str(source.get("producer") or ""),
                notes=str(entry.get("notes") or ""),
            )
        )
    return specs


def load_numbers(path: Path) -> List[NumberSpec]:
    """Load and validate the in-text numbers register.

    Args:
        path: Path to ``numbers.yaml``.

    Returns:
        One ``NumberSpec`` per entry, in file order.

    Raises:
        ValueError: If an entry is malformed or its source is neither a JSON
            ``key`` nor a CSV ``where`` + ``column``.
    """
    specs: List[NumberSpec] = []
    seen: Set[str] = set()
    for i, entry in enumerate(_read_yaml_list(path, "numbers")):
        where = f"{path.name} entry {i}"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: must be a mapping")
        num_id, used_in = _check_common(entry, _NUMBER_KEYS, where, seen)
        value, decimals = entry.get("value"), entry.get("decimals")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where} ({num_id}): 'value' must be a number")
        if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0:
            raise ValueError(f"{where} ({num_id}): 'decimals' must be an integer >= 0")
        source = entry.get("source")
        if not isinstance(source, dict) or set(source) - _NUMBER_SOURCE_KEYS:
            raise ValueError(f"{where} ({num_id}): malformed 'source'")
        src_path = source.get("path")
        if not isinstance(src_path, str):
            raise ValueError(f"{where} ({num_id}): source 'path' is required")
        is_json = src_path.endswith(".json")
        if is_json and not isinstance(source.get("key"), str):
            raise ValueError(f"{where} ({num_id}): a JSON source needs 'key'")
        if not is_json and not (
            src_path.endswith(".csv")
            and isinstance(source.get("where"), dict)
            and isinstance(source.get("column"), str)
        ):
            raise ValueError(
                f"{where} ({num_id}): source must be a .json with 'key' or a .csv "
                f"with 'where' and 'column'"
            )
        specs.append(
            NumberSpec(
                id=num_id,
                value=float(value),
                decimals=decimals,
                chapter=str(entry["chapter"]),
                used_in=used_in,
                path=src_path,
                key=source.get("key"),
                where={str(k): str(v) for k, v in (source.get("where") or {}).items()},
                column=source.get("column"),
                notes=str(entry.get("notes") or ""),
            )
        )
    return specs


# --------------------------------------------------------------------------- #
# Provenance helpers
# --------------------------------------------------------------------------- #


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(repo_root: Path) -> Tuple[str, bool]:
    """Return the ``HEAD`` commit and whether the working tree is dirty.

    Returns ``("unknown", True)`` when git is unavailable.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True
    return commit, bool(status)


def load_provenance(path: Path) -> Dict[str, BuildRecord]:
    """Load ``provenance.json`` (empty when the file does not exist)."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {asset_id: BuildRecord(**record) for asset_id, record in raw.items()}


def save_provenance(path: Path, records: Dict[str, BuildRecord]) -> None:
    """Write ``provenance.json`` sorted by asset id."""
    payload = {k: vars(records[k]) for k in sorted(records)}
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _rel(path: Path, repo_root: Path) -> str:
    """Repo-relative POSIX path."""
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def _resolve_render(target: str) -> Callable[[Path], Sequence[Path]]:
    """Import ``module:function``."""
    module_name, func_name = target.split(":", 1)
    func = getattr(importlib.import_module(module_name), func_name, None)
    if not callable(func):
        raise ValueError(f"render target {target!r} is not a callable")
    return func


def _render_code_files(target: str, repo_root: Path) -> List[Path]:
    """Source files of a render target that live in the repo.

    The module file and, for modules under ``thesis_assets.render``, the shared
    ``style.py``. Hashing them makes ``--check`` flag an asset as stale when its
    drawing code changes.
    """
    module = importlib.import_module(target.split(":", 1)[0])
    files: List[Path] = []
    origin = getattr(module, "__file__", None)
    if origin and Path(origin).resolve().is_relative_to(repo_root.resolve()):
        files.append(Path(origin).resolve())
    if module.__name__.startswith("thesis_assets.render."):
        style = Path(__file__).resolve().parent / "render" / "style.py"
        if style.resolve().is_relative_to(repo_root.resolve()) and style not in files:
            files.append(style)
    return files


def find_drawio() -> str:
    """Return the draw.io desktop CLI (``$DRAWIO_BIN`` first).

    Raises:
        FileNotFoundError: If no draw.io executable is found.
    """
    env = os.environ.get("DRAWIO_BIN")
    for candidate in ([env] if env else []) + list(_DRAWIO_CANDIDATES):
        found = shutil.which(candidate) or (
            candidate if Path(candidate).is_file() else None
        )
        if found:
            return found
    raise FileNotFoundError(
        "draw.io CLI not found; install draw.io desktop "
        "(brew install --cask drawio) or set DRAWIO_BIN"
    )


def export_drawio(src: Path, out_stem: Path) -> List[Path]:
    """Export a ``.drawio`` file to ``<out_stem>.svg`` and a 300-dpi PNG.

    Args:
        src: The ``.drawio`` source (first page is exported).
        out_stem: Output path without suffix.

    Returns:
        The SVG and PNG paths.

    Raises:
        RuntimeError: If the draw.io CLI fails or writes no file.
    """
    exe = find_drawio()
    jobs = [
        (out_stem.with_suffix(".svg"), ["--embed-svg-fonts", "false"]),
        (out_stem.with_suffix(".png"), ["--scale", str(_PNG_DPI / _SVG_DPI)]),
    ]
    for out, extra in jobs:
        cmd = [
            exe,
            "--export",
            "--theme",
            "light",
            "--border",
            "4",
            *extra,
            "--output",
            str(out),
        ]
        result = subprocess.run(
            cmd + [str(src)], capture_output=True, text=True, check=False
        )
        if result.returncode != 0 or not out.is_file():
            raise RuntimeError(
                f"draw.io export of {src} to {out.suffix} failed: "
                f"{result.stderr.strip()[-500:]}"
            )
    return [out for out, _ in jobs]


def build_asset(spec: AssetSpec, assets_dir: Path, repo_root: Path) -> BuildRecord:
    """Build one asset and return its provenance record.

    Old files of the same id are removed first, so a change of format leaves no
    stale file behind.

    Args:
        spec: The asset to build.
        assets_dir: The ``thesis_assets/`` folder.
        repo_root: Repository root; ``copy`` and ``inputs`` paths are relative
            to it.

    Returns:
        The build record (inputs and outputs hashed).

    Raises:
        FileNotFoundError: If a copy or drawio source or an input is missing.
        ValueError: If a render function writes nothing or writes outside
            ``<kind dir>/<id>.*``.
    """
    out_dir = assets_dir / KIND_DIRS[spec.kind]
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{spec.id}.*"):
        old.unlink()

    input_paths = [repo_root / p for p in spec.inputs]
    if spec.copy is not None:
        src = repo_root / spec.copy
        if not src.is_file():
            raise FileNotFoundError(f"{spec.id}: copy source {spec.copy} not found")
        dst = out_dir / f"{spec.id}{src.suffix}"
        shutil.copy2(src, dst)
        outputs = [dst]
        if src not in input_paths:
            input_paths.insert(0, src)
    elif spec.drawio is not None:
        src = repo_root / spec.drawio
        if not src.is_file():
            raise FileNotFoundError(f"{spec.id}: drawio source {spec.drawio} not found")
        outputs = export_drawio(src, out_dir / spec.id)
        if src not in input_paths:
            input_paths.insert(0, src)
    else:
        render = _resolve_render(str(spec.render))
        outputs = [Path(p) for p in render(out_dir / spec.id)]
        for code in _render_code_files(str(spec.render), repo_root):
            if code not in input_paths:
                input_paths.append(code)
        if not outputs:
            raise ValueError(f"{spec.id}: render {spec.render} wrote no files")
        for out in outputs:
            if out.resolve().parent != out_dir.resolve() or out.stem != spec.id:
                raise ValueError(
                    f"{spec.id}: render wrote {out}, expected {out_dir}/{spec.id}.*"
                )
            if not out.is_file():
                raise ValueError(f"{spec.id}: render returned missing file {out}")

    missing = [p for p in input_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{spec.id}: inputs not found: {missing}")
    commit, dirty = git_state(repo_root)
    return BuildRecord(
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=commit,
        git_dirty=dirty,
        inputs={_rel(p, repo_root): sha256_file(p) for p in input_paths},
        outputs={_rel(p, repo_root): sha256_file(p) for p in sorted(outputs)},
    )


def build(
    assets_dir: Path = ASSETS_DIR,
    repo_root: Path = REPO_ROOT,
    only: Optional[Sequence[str]] = None,
) -> List[str]:
    """Build the assets in the manifest and update ``provenance.json``.

    Args:
        assets_dir: The ``thesis_assets/`` folder.
        repo_root: Repository root.
        only: Ids to build; all assets when ``None``.

    Returns:
        The ids that were built.

    Raises:
        ValueError: If ``only`` names an id that is not in the manifest.
    """
    specs = load_manifest(assets_dir / MANIFEST_FILE)
    ids = [s.id for s in specs]
    if only:
        unknown = sorted(set(only) - set(ids))
        if unknown:
            raise ValueError(f"not in the manifest: {unknown}")
        specs = [s for s in specs if s.id in set(only)]
    prov_path = assets_dir / PROVENANCE_FILE
    records = load_provenance(prov_path)
    for spec in specs:
        records[spec.id] = build_asset(spec, assets_dir, repo_root)
        print(f"built {spec.id}")
    records = {k: v for k, v in records.items() if k in set(ids)}
    save_provenance(prov_path, records)
    return [s.id for s in specs]


# --------------------------------------------------------------------------- #
# Check
# --------------------------------------------------------------------------- #


def read_source_number(spec: NumberSpec, repo_root: Path) -> float:
    """Read the current value of a registered number from its source file.

    Raises:
        FileNotFoundError: If the source file does not exist.
        KeyError: If the JSON key or the CSV row/column cannot be found, or the
            CSV filter does not match exactly one row.
    """
    path = repo_root / spec.path
    if not path.is_file():
        raise FileNotFoundError(spec.path)
    if spec.key is not None:
        node: Any = json.loads(path.read_text(encoding="utf-8"))
        for part in spec.key.split("."):
            node = node[int(part)] if isinstance(node, list) else node[part]
        return float(node)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            r
            for r in csv.DictReader(handle)
            if all(r.get(k) == v for k, v in spec.where.items())
        ]
    if len(rows) != 1:
        raise KeyError(f"{spec.path}: filter {spec.where} matches {len(rows)} rows")
    if spec.column not in rows[0]:
        raise KeyError(f"{spec.path}: no column {spec.column!r}")
    return float(rows[0][str(spec.column)])


def check(assets_dir: Path = ASSETS_DIR, repo_root: Path = REPO_ROOT) -> CheckReport:
    """Validate the frozen assets against the manifest, provenance and inputs.

    Args:
        assets_dir: The ``thesis_assets/`` folder.
        repo_root: Repository root.

    Returns:
        Errors (missing, hand-edited, stale or orphan assets; numbers that no
        longer match) and warnings (inputs absent on this machine).
    """
    report = CheckReport()
    try:
        specs = load_manifest(assets_dir / MANIFEST_FILE)
        numbers = load_numbers(assets_dir / NUMBERS_FILE)
    except ValueError as exc:
        report.errors.append(str(exc))
        return report
    records = load_provenance(assets_dir / PROVENANCE_FILE)

    known_outputs: Set[str] = set()
    for spec in specs:
        record = records.get(spec.id)
        if record is None:
            report.errors.append(f"{spec.id}: never built")
            continue
        for out, digest in record.outputs.items():
            known_outputs.add(out)
            out_path = repo_root / out
            if not out_path.is_file():
                report.errors.append(f"{spec.id}: output {out} missing")
            elif sha256_file(out_path) != digest:
                report.errors.append(f"{spec.id}: output {out} changed since build")
        for inp, digest in record.inputs.items():
            inp_path = repo_root / inp
            if not inp_path.is_file():
                report.warnings.append(f"{spec.id}: input {inp} absent here")
            elif sha256_file(inp_path) != digest:
                report.errors.append(f"{spec.id}: stale, input {inp} changed")
        src_file = spec.copy or spec.drawio
        if src_file is not None and src_file not in record.inputs:
            report.errors.append(f"{spec.id}: source changed in manifest, rebuild")

    drawio_sources = {s.drawio for s in specs if s.drawio}
    diagram_dir = assets_dir / DIAGRAM_SOURCE_DIR
    if diagram_dir.is_dir():
        for path in sorted(diagram_dir.glob("*.drawio")):
            if _rel(path, repo_root) not in drawio_sources:
                report.errors.append(f"orphan diagram {_rel(path, repo_root)}")

    for kind_dir in sorted(set(KIND_DIRS.values())):
        folder = assets_dir / kind_dir
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                if _rel(path, repo_root) not in known_outputs:
                    report.errors.append(f"orphan file {_rel(path, repo_root)}")

    for num in numbers:
        try:
            current = read_source_number(num, repo_root)
        except FileNotFoundError:
            report.warnings.append(f"number {num.id}: source {num.path} absent here")
            continue
        except (KeyError, IndexError, ValueError) as exc:
            report.errors.append(f"number {num.id}: {exc}")
            continue
        if round(current, num.decimals) != round(num.value, num.decimals):
            report.errors.append(
                f"number {num.id}: text says {num.value}, source now gives "
                f"{round(current, num.decimals)}"
            )
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point; returns the process exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="+", metavar="ID", help="ids to build")
    parser.add_argument(
        "--check", action="store_true", help="validate only, build nothing"
    )
    args = parser.parse_args(argv)
    if args.check:
        report = check()
        for msg in report.warnings:
            print(f"warning: {msg}")
        for msg in report.errors:
            print(f"error: {msg}")
        print(f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)")
        return 1 if report.errors else 0
    build(only=args.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
