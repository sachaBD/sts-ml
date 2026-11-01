"""Optional end-to-end check: requires Playwright and its Chromium (not needed to build).

Serves only static files under a project subpath, without cross-origin isolation headers.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--simulations", type=int, default=4)
    parser.add_argument("--particles", type=int, default=2)
    parser.add_argument("--full-fight", action="store_true", help="also autoplay the first fight to completion")
    parser.add_argument("--all-fights", action="store_true", help="also search the initial position of every published fight")
    parser.add_argument("--native", type=Path, help="optional native nn_explain for initial-state/network parity")
    parser.add_argument("--checkpoint", type=Path, help="native model.onnx path")
    args = parser.parse_args()
    site = args.site.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    info = json.loads((site / "assets/gumbel/manifest.json").read_text())
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=site.parent))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    errors, api_calls = [], []
    report = {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("console", lambda message: print("browser:", message.type, message.text))
            page.on("request", lambda request: api_calls.append(request.url) if "/api/" in request.url else None)
            # No external resources (including Google Fonts) are required to run the page.
            page.route("**/*", lambda route: route.continue_() if route.request.url.startswith("http://127.0.0.1:") else route.abort())
            url = f"http://127.0.0.1:{server.server_port}/{site.name}/gumbel-tree/?sims={args.simulations}&particles={args.particles}&rollout_mix=0.5"
            page.goto(url)
            page.wait_for_function("/^(decision 0|error:)/.test(document.getElementById('status').textContent)", timeout=120000)
            status = page.locator("#status").inner_text()
            assert status.startswith("decision 0"), status
            assert page.locator("#root tr").count() > 1
            report["initial_status"] = status
            report["cross_origin_isolated"] = page.evaluate("crossOriginIsolated")
            assert report["cross_origin_isolated"] is False
            page.locator("#next").click()
            page.wait_for_function("/^(decision 1|error:)/.test(document.getElementById('status').textContent)", timeout=120000)
            assert page.locator("#status").inner_text().startswith("decision 1")
            report["next_status"] = page.locator("#status").inner_text()
            page.locator("#prev").click()
            page.wait_for_function("document.getElementById('status').textContent.startsWith('decision 0')")
            if args.full_fight:
                page.locator("#playFight").click()
                page.wait_for_function("/^(fight complete|error:)/.test(document.getElementById('status').textContent)", timeout=300000)
                report["full_fight"] = page.locator("#status").inner_text()
                assert report["full_fight"].startswith("fight complete"), report
            if args.all_fights:
                report["fight_statuses"] = []
                for fight in info["fights"]:
                    page.locator("#fight").select_option(fight["id"])
                    page.wait_for_function("/^(decision 0|error:)/.test(document.getElementById('status').textContent)", timeout=120000)
                    status = page.locator("#status").inner_text()
                    report["fight_statuses"].append({"fight": fight["id"], "status": status})
                    assert status.startswith("decision 0"), status
            page.screenshot(path=str(args.output / "gumbel.png"), full_page=True)
            # Root-only inference is directly comparable even when standard-library RNG samplers differ.
            result = page.evaluate("""async () => {
              const {browserAPI} = await import('./browser.js');
              const api = browserAPI();
              const info = await (await fetch('../assets/gumbel/manifest.json')).json();
              return api('decision', {run: info.run, iteration: info.iteration, fight: info.fights[0].id,
                moves: [], search: {sims: 0, particles: 2, gumbel_scale: 0, rollout_mix: 0, rescale_values: false}});
            }""")
            (args.output / "browser-root.json").write_text(json.dumps(result))
            assert not result.get("error"), result
            assert len(result["tree"]["nodes"]) == 1
            if args.native:
                if not args.checkpoint:
                    parser.error("--native needs --checkpoint")
                request = dict(model=str(args.checkpoint.resolve()), start=json.loads(info["fights"][0]["start_json"]), moves=[],
                               search=dict(sims=0, particles=2, gumbel_scale=0, rollout_mix=0, rescale_values=False))
                process = subprocess.run([str(args.native.resolve())], input=json.dumps(request) + "\n", text=True,
                                         capture_output=True, check=True, timeout=60)
                native = json.loads(process.stdout)
                (args.output / "native-root.json").write_text(json.dumps(native))
                assert native["view"] == result["view"], "native/WASM initial state differs"
                assert native["observation"] == result["observation"], "native/WASM observation key differs"
                a, b = native["tree"]["nodes"][0], result["tree"]["nodes"][0]
                assert [e["bits"] for e in a["edges"]] == [e["bits"] for e in b["edges"]]
                dv = abs(a["network"] - b["network"])
                dl = max(abs(x["logit"] - y["logit"]) for x, y in zip(a["edges"], b["edges"]))
                report.update(root_value_error=dv, root_max_logit_error=dl, root_actions=len(a["edges"]))
                assert dv < 1e-5 and dl < 1e-4, report
            assert not errors, errors
            assert not api_calls, api_calls
            report.update(page_errors=errors, backend_api_requests=api_calls)
            browser.close()
    finally:
        server.shutdown()
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
