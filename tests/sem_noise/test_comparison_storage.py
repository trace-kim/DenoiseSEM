import json

import numpy as np
import pytest

from sem_noise import comparison_storage as storage
from sem_noise.pipeline import _json_value, write_csv


def test_appended_contours_are_serialized_once_and_final_array_is_exact(tmp_path, monkeypatch):
    from sem_noise import pipeline

    first = {"coarse": [[1.25, 2.5], [-0.0, 3.0]], "status": {"coarse": "border"},
             "open_paths": [[[1., 2.], [3., 4.]]], "holes": [], "refined": [],
             "note": 'escaped "quote" and\nnewline', "missing": float("nan")}
    second = {"coarse": [[4.5, 6.75]], "refined": [[4.75, 6.875]], "refined_valid": [True],
              "holes": [[[1., 1.], [2., 2.], [3., 1.]]], "score": np.float64(.125)}
    expected = _json_value([first, second])
    site = {"name": "site", "contours": [first]}
    storage.checkpoint_contours(tmp_path, site)
    part = tmp_path / site["contours_parts"][0]["path"]
    first_bytes, first_mtime = part.read_bytes(), part.stat().st_mtime_ns
    convert = pipeline._json_value

    def no_reencode(value):
        if value is first:
            raise AssertionError("A completed contour was serialized again")
        return convert(value)

    monkeypatch.setattr(pipeline, "_json_value", no_reencode)
    storage.checkpoint_contours(tmp_path, site)  # An unchanged checkpoint does no work.
    site["contours"].append(second)
    storage.checkpoint_contours(tmp_path, site)
    assert [p["count"] for p in site["contours_parts"]] == [1, 1]
    snapshot = {k: v for k, v in site.items() if k != "contours"}
    assert storage.load_contours(tmp_path, snapshot) == expected
    storage.finish_contours(tmp_path, site)
    assert json.loads((tmp_path / site["contours_path"]).read_text()) == expected
    assert part.read_bytes() == first_bytes and part.stat().st_mtime_ns == first_mtime


def test_failed_part_does_not_publish_or_damage_previous_checkpoint(tmp_path):
    site = {"name": "site", "contours": [{"coarse": [[.5, 1.]]}]}
    storage.checkpoint_contours(tmp_path, site)
    snapshot = {"name": "site", "contours_parts": list(site["contours_parts"])}
    site["contours"].extend([{"coarse": [[2., 3.]]}, {"unsupported": object()}])
    with pytest.raises(TypeError):
        storage.checkpoint_contours(tmp_path, site)
    assert site["contours_parts"] == snapshot["contours_parts"]
    assert storage.load_contours(tmp_path, snapshot) == [{"coarse": [[.5, 1.]]}]
    assert len(list((tmp_path / "site/contour_parts").iterdir())) == 1


def test_failed_assembly_keeps_previous_final_file(tmp_path):
    site = {"name": "site", "contours": [{"coarse": [[1., 2.]]}]}
    storage.checkpoint_contours(tmp_path, site)
    previous = b'[{"previous":true}]\n'
    final = tmp_path / "site/contours.json"
    final.write_bytes(previous)
    (tmp_path / site["contours_parts"][0]["path"]).write_bytes(b"")
    with pytest.raises(ValueError, match="Incomplete contour checkpoint"):
        storage.finish_contours(tmp_path, site)
    assert final.read_bytes() == previous
    assert "contours_path" not in site
    assert not list(final.parent.glob(".*.tmp"))


def test_failed_metadata_write_keeps_previous_record(tmp_path):
    path = tmp_path / "comparison.json"
    storage.write_record(path, {"status": "running"})
    previous = path.read_bytes()
    with pytest.raises(TypeError):
        storage.write_record(path, {"bad": object()})
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob(".*.tmp"))


def test_old_contour_formats_and_empty_parts_are_readable(tmp_path):
    rows = [{"coarse": [[1.5, 2.5]]}]
    assert storage.load_contours(tmp_path, {"contours": rows}) == rows
    (tmp_path / "contours.json").write_text(json.dumps(rows))
    assert storage.load_contours(tmp_path, {"contours_path": "contours.json"}) == rows
    site = {"name": "empty", "contours": []}
    storage.finish_contours(tmp_path, site)
    assert storage.load_contours(tmp_path, {"contours_path": site["contours_path"]}) == []
    for bad in ({"contours_path": "../elsewhere.json"},
                {"contours_parts": [{"path": "../elsewhere.jsonl", "count": 1}]}):
        with pytest.raises(ValueError, match="inside its directory"):
            storage.load_contours(tmp_path, bad)
    with pytest.raises(ValueError, match="Missing saved contours"):
        storage.load_contours(tmp_path, {"name": "site"})


def test_json_and_streamed_csv_keep_nonfinite_and_numpy_conversion(tmp_path):
    values = {"python": [1.25, float("nan"), float("inf"), float("-inf")],
              "numpy": np.array([1.25, np.nan, np.inf]), "bool": np.bool_(True),
              "int": np.int64(7), "tuple": (np.float32(.5), None)}
    expected = {"python": [1.25, None, None, None], "numpy": [1.25, None, None],
                "bool": True, "int": 7, "tuple": [.5, None]}
    assert _json_value(values) == expected
    path = tmp_path / "table.csv"
    write_csv(path, [{"value": np.float64(1.25)}, {"value": float("nan"), "extra": np.int64(2)}])
    assert path.read_text().splitlines() == ["value,extra", "1.25,", ",2"]
