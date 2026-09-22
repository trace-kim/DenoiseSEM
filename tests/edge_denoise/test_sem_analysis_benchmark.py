import json

from test_real_sem_viewer import saved_record
from tools import benchmark_sem_analysis as benchmark


def test_remote_benchmark_uses_saved_sources_and_reports_agreement(tmp_path, fake_cupy):
    source = saved_record(tmp_path / "report", count=3)
    before = {p.relative_to(source.parent): p.read_bytes() for p in source.parent.rglob("*") if p.is_file()}
    result = benchmark.benchmark(source, device="cuda:1", frames_per_source=3, batch_size=2)
    assert result["passed"]
    assert {r["source"] for r in result["sources"]} == {"raw", "model", "average8"}
    assert all(r["images_per_s"] > 0 and not r["differences"] for r in result["sources"])
    assert result["sources"][0]["warmup_s"] > 0
    assert {p.relative_to(source.parent): p.read_bytes() for p in source.parent.rglob("*") if p.is_file()} == before
    output = tmp_path / "benchmark.json"
    assert benchmark.main(["--from-comparison", str(source), "--device", "cpu", "--frames-per-source", "1",
                           "--output-json", str(output)]) == 0
    assert json.loads(output.read_text())["passed"]


def test_benchmark_fails_loudly_on_a_mask_or_threshold_difference(tmp_path, fake_cupy, monkeypatch):
    from sem_segment import otsu_cuda

    source = saved_record(tmp_path / "report", count=2)
    actual = otsu_cuda.cuda_masks

    def broken(*args, **kwargs):
        labels, thresholds, counts, retained, timings = actual(*args, **kwargs)
        labels[:, 0, 0] = 1
        return labels, thresholds + 2., counts, retained, timings

    monkeypatch.setattr(otsu_cuda, "cuda_masks", broken)
    result = benchmark.benchmark(source, frames_per_source=1)
    assert not result["passed"]
    assert all("mask" in s["differences"][0]["fields"] and "threshold" in s["differences"][0]["fields"]
               for s in result["sources"])
    assert benchmark.main(["--from-comparison", str(source), "--frames-per-source", "1"]) == 1
