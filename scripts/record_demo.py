import io
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

OUT = Path("assets/demo.gif")
OUT.parent.mkdir(parents=True, exist_ok=True)
SIZE = {"width": 1050, "height": 720}
URL = "http://127.0.0.1:7861"

frames: list[tuple[Image.Image, int]] = []

def shot(page, ms: int) -> None:
    page.wait_for_timeout(400)
    img = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
    frames.append((img, ms))

def main():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    proc = subprocess.Popen([sys.executable, "demo/app.py"], env=env)
    
    # Wait for Gradio to be ready
    ready = False
    for _ in range(30):
        time.sleep(1)
        try:
            with urllib.request.urlopen(URL, timeout=1) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            pass
            
    if not ready:
        proc.terminate()
        proc.wait()
        raise RuntimeError("Gradio failed to start on port 7861")
        
    print("Gradio server is ready. Recording frames...")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport=SIZE, color_scheme="light")
            page.goto(URL, timeout=30000)
            page.wait_for_selector(".model-card", timeout=15000)
            
            shot(page, 2000)
            
            # Scroll down to show all model cards clearly
            page.mouse.wheel(0, 320)
            shot(page, 2000)
            
            # Pick Pens Purchase
            dropdown = page.get_by_label("Select a Benchmark Example")
            dropdown.click()
            page.get_by_role("option", name="Pens Purchase (Simple Multiplication)").click()
            page.wait_for_timeout(1000)
            shot(page, 2500)
            
            # Pick Farm Animals
            dropdown.click()
            page.get_by_role("option", name="Farm Animals (Multi-step Addition)").click()
            page.wait_for_timeout(1000)
            shot(page, 2500)
            
            browser.close()
    finally:
        proc.terminate()
        proc.wait()
        
    print(f"Captured {len(frames)} frames. Encoding GIF...")
    images = [f.resize((840, 576), Image.LANCZOS).quantize(colors=128, method=Image.MEDIANCUT)
              for f, _ in frames]
    images[0].save(OUT, save_all=True, append_images=images[1:], loop=0, optimize=True,
                   duration=[ms for _, ms in frames])
    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"Saved {OUT}: {len(frames)} frames, {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
