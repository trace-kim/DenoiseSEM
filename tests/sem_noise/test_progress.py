import threading

import pytest

from sem_noise.progress import Progress


def test_blocking_stage_emits_heartbeat_without_item_progress(capsys, monkeypatch):
    heartbeat = threading.Event()
    timings = {}
    progress = Progress("large JSON write", timings=timings, key="save", interval_s=.01)
    emit = progress._emit

    def capture(message):
        emit(message)
        if "still running" in message:
            heartbeat.set()

    monkeypatch.setattr(progress, "_emit", capture)
    with progress:
        assert heartbeat.wait(timeout=2), "No heartbeat during a blocking operation"
    output = capsys.readouterr().out
    assert output.index("starting") < output.index("still running") < output.index("complete in")
    assert timings["save"] > 0
    assert not progress._thread.is_alive()


def test_failed_stage_reports_error_and_stops_heartbeat(capsys):
    progress = Progress("contours.json")
    with pytest.raises(OSError, match="disk full"):
        with progress:
            raise OSError("disk full")
    output = capsys.readouterr().out
    assert "starting" in output and "failed after" in output and "disk full" in output
    assert "complete in" not in output
    assert not progress._thread.is_alive()
