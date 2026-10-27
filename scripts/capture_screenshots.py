"""Capture the README screenshots from a running harness at http://127.0.0.1:8765.

Loads the NPU model, sends a few real requests so the charts have data, then screenshots each tab
with Playwright driving the installed Microsoft Edge (no browser download):

    uv run --with playwright python scripts/capture_screenshots.py docs/screenshots

Check the images before committing: they show your hardware, driver versions and process names.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8765"
NPU_MODEL = "qwen2.5-0.5b-instruct-openvino-npu:5"
CPU_MODEL = "qwen2.5-0.5b-instruct-generic-cpu:4"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


def call(path, body=None, method=None):
    req = urllib.request.Request(BASE + path, method=method or ("POST" if body is not None else "GET"),
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read() or b"null")


def wait_jobs():
    while any(j["state"] == "running" for j in call("/api/jobs")):
        time.sleep(1)


# Clean, NPU-focused state: only the NPU model loaded, with a fresh load notice.
for m in call("/api/models"):
    if m["loaded"]:
        call(f"/api/models/unload/{m['id']}", {})
wait_jobs()
call(f"/api/models/load/{NPU_MODEL}", {})
wait_jobs()

# Some real NPU traffic so the charts and request table have data.
prompts = ["Explain what an NPU is in two sentences.", "Write a haiku about silicon.",
           "List three uses for on-device AI.", "Summarise why low-power inference matters."]
for p in prompts:
    call("/v1/chat/completions", {"model": NPU_MODEL, "max_tokens": 120,
                                  "messages": [{"role": "user", "content": p}]})
    time.sleep(2)

with sync_playwright() as pw:
    browser = pw.chromium.launch(channel="msedge")
    for scheme in ("light", "dark"):
        ctx = browser.new_context(viewport={"width": 1360, "height": 900}, device_scale_factor=2, color_scheme=scheme)
        page = ctx.new_page()
        suffix = "" if scheme == "light" else "-dark"

        page.goto(f"{BASE}/#dashboard")
        page.wait_for_timeout(5000)  # let the live stream and driver details arrive
        page.screenshot(path=OUT / f"dashboard{suffix}.png", full_page=True)
        if scheme == "dark":
            ctx.close()
            continue

        page.goto(f"{BASE}/#models")
        page.wait_for_timeout(3000)
        page.screenshot(path=OUT / "models.png")

        page.goto(f"{BASE}/#playground")
        page.wait_for_timeout(2000)
        page.select_option("#chat-model", NPU_MODEL)
        page.fill("#temp", "0")
        page.fill("#chat-input", "Write a haiku about a tiny chip that thinks.")
        page.press("#chat-input", "Enter")
        page.wait_for_function("document.querySelector('#chat-send').textContent === 'Send'", timeout=120000)
        page.wait_for_timeout(500)
        page.screenshot(path=OUT / "playground.png")

        page.goto(f"{BASE}/#api")
        page.wait_for_timeout(1500)
        page.screenshot(path=OUT / "api.png")
        ctx.close()
    browser.close()

for f in sorted(OUT.glob("*.png")):
    print(f.name, f.stat().st_size // 1024, "KB")
