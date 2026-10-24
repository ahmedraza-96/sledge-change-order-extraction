import json

from co_extract import history

RESULT = {
    "document_confidence": 0.82,
    "needs_review": ["owner", "subtotal"],
    "issues": [],
    "meta": {"provider": "openrouter", "model": "nvidia/nemotron-3.5-super-vl:free"},
}


def test_record_and_read_back():
    run_id = history.record(filename="co.pdf", source="file", document=b"%PDF-1.7 ...", provider="openrouter",
                            result=RESULT, duration_ms=1234)
    assert run_id is not None
    [summary] = history.list_runs()
    assert summary["id"] == run_id
    assert summary["status"] == "ok" and summary["has_document"]
    assert summary["model"] == "nvidia/nemotron-3.5-super-vl:free"
    assert summary["review_count"] == 2 and summary["is_change_order"] is True
    assert "result" not in summary and "document" not in summary
    run = history.get_run(run_id)
    assert run["result"] == json.loads(json.dumps(RESULT))
    assert history.get_document(run_id) == ("co.pdf", b"%PDF-1.7 ...")


def test_failed_run_is_recorded():
    run_id = history.record(filename="pasted.txt", source="text", document=b"hello", provider="openrouter",
                            error="rate limited")
    run = history.get_run(run_id)
    assert run["status"] == "error" and run["error"] == "rate limited"
    assert run["result"] is None and run["document_confidence"] is None


def test_not_a_change_order_is_marked():
    result = {**RESULT, "issues": [{"code": "not_a_change_order"}]}
    run_id = history.record(filename="invoice.pdf", source="file", document=b"x", provider="demo", result=result)
    assert history.get_run(run_id)["is_change_order"] is False


def test_newest_first_and_paging():
    ids = [history.record(filename=f"{i}.txt", source="text", document=b"x", provider="demo", result=RESULT)
           for i in range(5)]
    assert [r["id"] for r in history.list_runs()] == ids[::-1]
    assert [r["id"] for r in history.list_runs(limit=2)] == ids[:2:-1]
    assert [r["id"] for r in history.list_runs(limit=2, before=ids[3])] == [ids[2], ids[1]]


def test_oversized_document_keeps_the_result_only(monkeypatch):
    monkeypatch.setattr(history, "MAX_DOCUMENT_BYTES", 10)
    run_id = history.record(filename="big.pdf", source="file", document=b"x" * 11, provider="demo", result=RESULT)
    assert history.get_run(run_id)["has_document"] is False
    assert history.get_run(run_id)["result"] is not None
    assert history.get_document(run_id) is None


def test_off_records_nothing(monkeypatch):
    monkeypatch.setenv("HISTORY_DB", "off")
    assert history.record(filename="a.txt", source="text", document=b"x", provider="demo", result=RESULT) is None
    assert history.list_runs() == []
    assert history.get_run(1) is None


def test_broken_path_does_not_raise(tmp_path, monkeypatch):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("a file where a folder should be")
    monkeypatch.setenv("HISTORY_DB", str(blocker / "history.db"))
    assert history.record(filename="a.txt", source="text", document=b"x", provider="demo", result=RESULT) is None


def test_missing_run():
    assert history.get_run(999) is None
    assert history.get_document(999) is None


def test_delete_run():
    run_id = history.record(filename="a.txt", source="text", document=b"x", provider="demo", result=RESULT)
    assert history.delete_run(run_id) is True
    assert history.get_run(run_id) is None and history.get_document(run_id) is None
    assert history.delete_run(run_id) is False
