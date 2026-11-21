# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright==1.63.0"]
# ///
"""Exercise the shopper flow against a running internal storefront in a fresh Chromium profile.

    uvx --from playwright==1.63.0 playwright install chromium
    uv run scripts/smoke_storefront.py [--url http://127.0.0.1:8765/]
"""
import argparse
import re
from time import monotonic
from urllib.parse import parse_qs, urlsplit

from playwright.sync_api import expect, sync_playwright

QUERY = "dark chocolate gift box for a friend"
BROWSER_MODEL = re.compile(
    r"\.(?:gguf|onnx|wasm|safetensors)(?:/|$)|/(?:model\.json|llm\.js|wllama[^/]*|api/(?:intent|model))(?:/|$)",
    re.I,
)


def check(url: str) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.set_default_timeout(30_000)
        requests, errors, prompts = [], [], []
        page.on("request", lambda request: requests.append(request.url))
        page.on("pageerror", lambda error: errors.append(str(error)))

        def dismiss(dialog):
            prompts.append(dialog.message)
            dialog.dismiss()

        page.on("dialog", dismiss)

        def run(query, route, action):
            start = monotonic()

            def is_query(response):
                path = urlsplit(response.url)
                return path.path.endswith("/api/search") and parse_qs(path.query).get("q") == [query]

            with page.expect_response(is_query) as event:
                action()
            response = event.value
            assert response.ok, f"{response.status} for {query}"
            result = response.json()
            assert result["query"] == query and result["route"] == route, result
            expect(page.locator("#result-query")).to_contain_text(query)
            expect(page.locator("#route")).to_have_text(
                "Teacher cache" if route == "teacher-cache" else "Live student"
            )
            ids = [p["id"] for p in result["products"]]
            item = "64021" if route == "teacher-cache" else "4188"
            assert item in ids, (query, ids)
            expect(page.locator("#products")).to_contain_text(f"Source item {item}")
            if route == "live-student":
                assert "lora" in result["backend"].lower(), result["backend"]
                assert "browser" not in result["backend"].lower(), result["backend"]
            print(f"{query!r} → {route}, {result['backend']}, item {item}, "
                  f"{len(response.body()):,} response bytes, {monotonic() - start:.2f} s")

        run("Mooncakes", "teacher-cache", lambda: page.goto(url))
        run(QUERY, "live-student", lambda: page.get_by_role("searchbox").fill(QUERY))
        assert parse_qs(urlsplit(page.url).query).get("q") == [QUERY], page.url
        run(QUERY, "live-student", page.reload)
        expect(page.get_by_role("searchbox")).to_have_value(QUERY)
        forbidden = [u for u in requests if BROWSER_MODEL.search(urlsplit(u).path)]
        assert not forbidden, forbidden
        assert not errors and not prompts, (errors, prompts)
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765/", help="Running internal storefront")
    check(parser.parse_args().url)
