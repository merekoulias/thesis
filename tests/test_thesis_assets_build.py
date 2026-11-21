"""Contract tests for ``thesis_assets.build`` on a tiny fake repo."""

import json
from pathlib import Path
from typing import List

import pytest
import yaml

from thesis_assets import build as tab


def _setup(tmp_path: Path, assets: List[dict], numbers: List[dict] = ()) -> Path:
    """Create ``<tmp>/repo`` with an artifact and a ``thesis_assets`` folder."""
    repo = tmp_path / "repo"
    art = repo / "artifacts"
    art.mkdir(parents=True)
    (art / "fig.png").write_bytes(b"png-bytes")
    (art / "metrics.json").write_text(json.dumps({"auc": 0.78621, "f": [{"x": 2}]}))
    (art / "utility.csv").write_text("model,tpr\nlgbm,0.32507\nrf,0.3\n")
    assets_dir = repo / "thesis_assets"
    assets_dir.mkdir()
    (assets_dir / "manifest.yaml").write_text(yaml.safe_dump({"assets": assets}))
    (assets_dir / "numbers.yaml").write_text(yaml.safe_dump({"numbers": list(numbers)}))
    return repo


def _asset(**overrides: object) -> dict:
    entry = {
        "id": "primary-gaps",
        "kind": "figure",
        "chapter": "6.2",
        "caption": "Σχήμα: κύριες διαφορές",
        "used_in": ["thesis"],
        "source": {"copy": "artifacts/fig.png"},
    }
    entry.update(overrides)
    return entry


def _build(repo: Path) -> List[str]:
    return tab.build(assets_dir=repo / "thesis_assets", repo_root=repo)


def _check(repo: Path) -> tab.CheckReport:
    return tab.check(assets_dir=repo / "thesis_assets", repo_root=repo)


def test_copy_build_then_check_is_clean(tmp_path: Path) -> None:
    repo = _setup(tmp_path, [_asset()])
    assert _build(repo) == ["primary-gaps"]
    out = repo / "thesis_assets" / "figures" / "primary-gaps.png"
    assert out.read_bytes() == b"png-bytes"
    prov = json.loads((repo / "thesis_assets" / "provenance.json").read_text())
    assert list(prov["primary-gaps"]["inputs"]) == ["artifacts/fig.png"]
    report = _check(repo)
    assert report.errors == [] and report.warnings == []


def test_check_flags_stale_input_hand_edit_and_orphan(tmp_path: Path) -> None:
    repo = _setup(tmp_path, [_asset()])
    _build(repo)
    (repo / "artifacts" / "fig.png").write_bytes(b"regenerated")
    (repo / "thesis_assets" / "figures" / "primary-gaps.png").write_bytes(b"edited")
    (repo / "thesis_assets" / "figures" / "stray.png").write_bytes(b"x")
    errors = " | ".join(_check(repo).errors)
    assert "stale" in errors
    assert "changed since build" in errors
    assert "orphan file thesis_assets/figures/stray.png" in errors


def test_missing_input_is_a_warning_and_unbuilt_is_an_error(tmp_path: Path) -> None:
    repo = _setup(tmp_path, [_asset()])
    _build(repo)
    (repo / "artifacts" / "fig.png").unlink()
    report = _check(repo)
    assert report.errors == [] and len(report.warnings) == 1
    repo2 = _setup(tmp_path / "b", [_asset()])
    assert _check(repo2).errors == ["primary-gaps: never built"]


def test_render_writes_under_id_and_removes_old_formats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_render(out_stem: Path) -> List[Path]:
        paths = [out_stem.with_suffix(".pdf"), out_stem.with_suffix(".png")]
        for p in paths:
            p.write_bytes(b"r")
        return paths

    monkeypatch.setattr(tab, "_resolve_render", lambda target: fake_render)
    monkeypatch.setattr(tab, "_render_code_files", lambda target, root: [])
    repo = _setup(
        tmp_path,
        [_asset(source={"render": "x:y", "inputs": ["artifacts/metrics.json"]})],
    )
    (repo / "thesis_assets" / "figures").mkdir()
    (repo / "thesis_assets" / "figures" / "primary-gaps.svg").write_bytes(b"old")
    _build(repo)
    names = sorted(p.name for p in (repo / "thesis_assets" / "figures").iterdir())
    assert names == ["primary-gaps.pdf", "primary-gaps.png"]
    assert _check(repo).errors == []


def test_render_outside_its_stem_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def bad_render(out_stem: Path) -> List[Path]:
        other = out_stem.parent / "other.png"
        other.write_bytes(b"r")
        return [other]

    monkeypatch.setattr(tab, "_resolve_render", lambda target: bad_render)
    monkeypatch.setattr(tab, "_render_code_files", lambda target, root: [])
    repo = _setup(tmp_path, [_asset(source={"render": "x:y"})])
    with pytest.raises(ValueError, match="expected"):
        _build(repo)


@pytest.mark.parametrize(
    "bad",
    [
        [_asset(), _asset()],
        [_asset(id="Figure 4.2")],
        [_asset(kind="chart")],
        [_asset(used_in=["paper"])],
        [_asset(caption="")],
        [_asset(source={"copy": "a", "render": "m:f"})],
        [_asset(source={})],
        [_asset(extra=1)],
        [_asset(kind="diagram")],
        [_asset(source={"drawio": "thesis_assets/diagrams/a.drawio"})],
        [_asset(kind="diagram", source={"drawio": "a.svg"})],
    ],
)
def test_manifest_validation(tmp_path: Path, bad: List[dict]) -> None:
    repo = _setup(tmp_path, bad)
    with pytest.raises(ValueError):
        tab.load_manifest(repo / "thesis_assets" / "manifest.yaml")


def test_drawio_source_is_hashed_and_orphan_diagrams_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_export(src: Path, out_stem: Path) -> List[Path]:
        out = out_stem.with_suffix(".svg")
        out.write_text(src.read_text())
        return [out]

    monkeypatch.setattr(tab, "export_drawio", fake_export)
    drawio = "thesis_assets/diagrams/flow.drawio"
    repo = _setup(
        tmp_path, [_asset(id="flow", kind="diagram", source={"drawio": drawio})]
    )
    (repo / "thesis_assets" / "diagrams").mkdir()
    (repo / drawio).write_text("<mxfile/>")
    _build(repo)
    assert (repo / "thesis_assets" / "figures" / "flow.svg").is_file()
    assert _check(repo).errors == []
    (repo / drawio).write_text("<mxfile>edited</mxfile>")
    (repo / "thesis_assets" / "diagrams" / "old.drawio").write_text("<mxfile/>")
    errors = " | ".join(_check(repo).errors)
    assert "stale, input thesis_assets/diagrams/flow.drawio changed" in errors
    assert "orphan diagram thesis_assets/diagrams/old.drawio" in errors


def test_render_code_files_include_module_and_style() -> None:
    files = tab._render_code_files(
        "thesis_assets.render.ch2:roc_illustration", tab.REPO_ROOT
    )
    assert [f.name for f in files] == ["ch2.py", "style.py"]


def test_greek_number_and_figure_width(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    from thesis_assets.render.style import cm, greek_number, save_figure

    assert greek_number(-0.25, 2) == "\u22120,25"
    assert greek_number(1, 1) == "1,0"
    fig = plt.figure(figsize=(cm(17.0), cm(5.0)))
    with pytest.raises(ValueError, match="max 16.0 cm"):
        save_figure(fig, tmp_path / "too-wide")
    plt.close(fig)


def _number(**overrides: object) -> dict:
    entry = {
        "id": "lgbm-auc",
        "value": 0.786,
        "decimals": 3,
        "chapter": "6.1",
        "used_in": ["thesis"],
        "source": {"path": "artifacts/metrics.json", "key": "auc"},
    }
    entry.update(overrides)
    return entry


def test_numbers_match_and_mismatch(tmp_path: Path) -> None:
    numbers = [
        _number(),
        _number(
            id="nested",
            value=2,
            decimals=0,
            source={"path": "artifacts/metrics.json", "key": "f.0.x"},
        ),
        _number(
            id="tpr",
            value=0.325,
            source={
                "path": "artifacts/utility.csv",
                "where": {"model": "lgbm"},
                "column": "tpr",
            },
        ),
        _number(id="wrong", value=0.79),
        _number(
            id="ambiguous",
            source={"path": "artifacts/utility.csv", "where": {}, "column": "tpr"},
        ),
    ]
    repo = _setup(tmp_path, [], numbers)
    errors = _check(repo).errors
    assert len(errors) == 2
    assert errors[0].startswith("number wrong: text says 0.79")
    assert "matches 2 rows" in errors[1]


def test_number_validation(tmp_path: Path) -> None:
    repo = _setup(tmp_path, [], [_number(source={"path": "artifacts/utility.csv"})])
    with pytest.raises(ValueError, match="csv"):
        tab.load_numbers(repo / "thesis_assets" / "numbers.yaml")


def test_repo_manifest_and_numbers_are_valid() -> None:
    tab.load_manifest(tab.ASSETS_DIR / tab.MANIFEST_FILE)
    tab.load_numbers(tab.ASSETS_DIR / tab.NUMBERS_FILE)


def test_save_table_writes_exact_csv_and_greek_html(tmp_path: Path) -> None:
    import pandas as pd

    from thesis_assets.render.style import save_table

    table = pd.DataFrame({"Μοντέλο": ["LR", "GBDT"], "AUC": [0.776699, 0.786206]})
    csv_path, html_path = save_table(table, tmp_path / "auc", decimals=3)
    assert pd.read_csv(csv_path)["AUC"].tolist() == [0.776699, 0.786206]
    html = html_path.read_text(encoding="utf-8")
    assert "0,777" in html and "0,786" in html and "Times New Roman" in html
