from fastapi.testclient import TestClient


def make_client(tmp_path, monkeypatch):
    from pipeline import config, gui
    cfg = config.load()
    cfg["paths"]["workspace"] = str(tmp_path / "ws")
    cfg["paths"]["db"] = str(tmp_path / "ws/p.db")
    monkeypatch.setattr(gui, "cfg", cfg)
    monkeypatch.setattr(gui, "_conn", None)
    # never launch real models from a unit test
    monkeypatch.setattr(gui, "_run_in_background", lambda run_id: None)
    return TestClient(gui.app)


def test_page_and_upload_creates_run(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    assert "instruction.md" in client.get("/").text

    r = client.post("/api/runs", files=[
        ("instruction", ("instruction.md", b"# a tiny fighter", "text/markdown")),
        ("refs", ("style.png", b"\x89PNGfake", "image/png")),
    ])
    assert r.status_code == 200
    run_id = r.json()["run_id"]

    runs = client.get("/api/runs").json()
    assert runs[0]["id"] == run_id
    assert "style.png" in runs[0]["reference_images"]
    assert client.get(f"/api/runs/{run_id}/tasks").json() == []
