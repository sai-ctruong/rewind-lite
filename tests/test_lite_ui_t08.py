"""The Lite demo is served by the same Flask app as its real API."""

from pathlib import Path

from ui.lite_app import create_lite_app


def test_lite_demo_entry_and_assets_are_served_without_legacy_features(tmp_path):
    app = create_lite_app(video_dir=tmp_path / "videos",
                          index_root=tmp_path / "indexes")
    client = app.test_client()

    page = client.get("/")
    assert page.status_code == 200
    assert page.mimetype == "text/html"
    html = page.get_data(as_text=True)
    assert 'lang="vi"' in html
    assert 'id="video-player"' in html
    assert 'id="search-form"' in html
    assert 'id="results"' in html
    assert "/api/lite/" in client.get("/lite.js").get_data(as_text=True)
    assert client.get("/lite.css").mimetype == "text/css"
    assert client.get("/lite.js").mimetype in {"text/javascript", "application/javascript"}
    assert client.post("/api/video/vqa", json={}).status_code == 404
    assert client.get("/api/agent/ask").status_code == 404
