"""Browser regression for static-frame evidence and explicit UI state.

UI01_BASELINE=1 replays the pre-change UI to demonstrate failure.
"""
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pytest

sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def test_ui01_evidence_identity_and_layout():
    with sync_playwright() as p:
        if not Path(p.chromium.executable_path).is_file():
            pytest.skip("UI01 browser test needs a locally installed Chromium")
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        posts = []
        fid = "v/1"
        base = dict(frame_id=fid, timestamp=10., bbox_norm_xyxy=[.1,.2,.5,.8], class_name="person", confidence=.9)
        hit = dict(rank=1, score=1.1, score_type="temporal_object_reranked", video_id="v",
                   primary_frame_id=fid, peak_timestamp=10., t_start=8., t_end=12.,
                   image_url="/frame.jpg", detections=[base, {**base,"frame_id":"v/other"}, {**base,"timestamp":11.}])
        _, jpeg = cv2.imencode('.jpg',np.full((100,200,3),128,dtype=np.uint8))
        vid = "a"*32
        def route(r):
            path = r.request.url.split('http://ui.test')[-1]
            if path.startswith('/api/'):
                if r.request.method == 'POST': posts.append(path)
                data = {'videos':[vid], 'cached':True,'frames':1,'results':[hit], 'mode':'temporal_object'}
                r.fulfill(json=data)
            elif path == '/frame.jpg': r.fulfill(body=jpeg.tobytes(),content_type='image/jpeg')
            else:
                file = {'/':'ui/lite_index.html','/lite.js':'ui/lite.js','/lite.css':'ui/lite.css'}.get(path)
                if not file: return r.fulfill(status=404)
                body = (subprocess.check_output(['git','show','754e07f:'+file]) if os.getenv('UI01_BASELINE')
                        else (ROOT/file).read_bytes())
                r.fulfill(body=body,content_type={'.js':'text/javascript','.css':'text/css'}.get(Path(file).suffix,'text/html'))
        page.route('http://ui.test/**',route)
        page.goto('http://ui.test/')
        page.wait_for_function("!document.querySelector('#search-query').disabled")
        assert not page.locator('#toggle-yolo').is_checked()
        page.locator('#toggle-yolo').check()
        page.locator('#search-query').fill('two people')
        page.locator('#search-button').click()
        page.wait_for_selector('.result-button')
        assert page.locator('#toggle-bbox').count() == 0
        assert page.locator('.evidence-box').count() == 0
        page.evaluate('Object.defineProperty(document.querySelector("#video-player"),"duration",{value:100}); document.querySelector("#video-player").dispatchEvent(new Event("loadedmetadata"))')
        assert page.locator('.interval').count() == 1
        assert page.locator('.interval').evaluate('(e)=>e.style.width') == '4%'
        page.evaluate('Object.defineProperty(document.querySelector("#video-player"),"readyState",{value:1})')
        page.locator('.interval').focus(); page.keyboard.press('Enter')
        assert page.locator('#video-player').evaluate('(v)=>v.currentTime') == 8
        page.locator('#video-player').evaluate('(v)=>{v.currentTime=12.3;v.dispatchEvent(new Event("timeupdate"));}')
        assert page.locator('#video-player').evaluate('(v)=>v.paused && v.currentTime===12')
        for width in (1440,390):
            page.set_viewport_size({'width':width,'height':844})
            if width == 390: page.locator('#menu-toggle').click()
            assert page.locator('.video-name').evaluate('(e)=>e.getBoundingClientRect().right <= document.querySelector("#sidebar").getBoundingClientRect().right')
            if width == 390: page.keyboard.press('Escape')
        page.locator('#search-query').fill('new query')
        assert page.locator('.evidence-box').count() == 0
        assert page.locator('.interval').count() == 0
        hit['detections'] = None
        page.locator('#search-button').click()
        page.wait_for_selector('.result-button')
        assert page.locator('.evidence-box').count() == 0
        browser.close()
