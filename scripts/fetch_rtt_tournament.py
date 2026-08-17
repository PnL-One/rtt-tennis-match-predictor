from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rtt_predictor.tournament_data import (
    RTT_PUBLIC_ROOT,
    TOURNAMENT_ROUTES,
    build_snapshot_from_pages,
    read_cached_pages,
    save_snapshot,
)


DEFAULT_CACHE_DIR = PROJECT_ROOT / "tournament_analysis_cache"
RTT_API_HEALTH_URL = "https://apirtt.mytennis.online/api/v1/auth/login"


_PLAYER_PANEL_HEADINGS = {
    "requests": ("основной турнир", "ожидающие игроки"),
    "members": ("основной турнир", "квалификационный турнир"),
}


def ensure_playwright_interpreter() -> None:
    """Re-run under stable CPython with project packages when Playwright is absent."""

    if importlib.util.find_spec("playwright") is not None:
        return
    project_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))
    stable_python = local_app_data / "Programs" / "Python" / "Python312" / "python.exe"
    target_python = stable_python if stable_python.exists() else project_python
    already_reexecuted = os.environ.get("RTT_FETCH_REEXECUTED") == "1"
    if not target_python.exists() or already_reexecuted:
        raise SystemExit(
            "Playwright is not installed in either the current Python or the project .venv. "
            "Install project requirements first."
        )
    env = os.environ.copy()
    env["RTT_FETCH_REEXECUTED"] = "1"
    venv_site_packages = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
    if venv_site_packages.exists():
        previous_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(venv_site_packages) + (
            os.pathsep + previous_pythonpath if previous_pythonpath else ""
        )
    browser_cache = PROJECT_ROOT / "tmp" / "ms-playwright"
    if browser_cache.exists():
        env.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(browser_cache))
    completed = subprocess.run(
        [str(target_python), "-S", str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=PROJECT_ROOT,
        env=env,
    )
    raise SystemExit(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh isolated pages for RTT tournament simulation.")
    parser.add_argument("--tour-id", action="append", required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=45)
    parser.add_argument("--browser", choices=("firefox", "chromium"), default="firefox")
    parser.add_argument(
        "--route",
        action="append",
        choices=tuple(route.rstrip("/") for route in TOURNAMENT_ROUTES),
        help="Refresh only selected route(s); intended for diagnostics.",
    )
    return parser


async def wait_until_stable(page, timeout_ms: int) -> None:
    previous = (-1, -1)
    stable_rounds = 0
    deadline = asyncio.get_running_loop().time() + timeout_ms / 1000
    while asyncio.get_running_loop().time() < deadline:
        current = (
            await page.locator("table tbody tr").count(),
            len(await page.locator("body").inner_text()),
        )
        if current == previous and current[1] > 100:
            stable_rounds += 1
            if stable_rounds >= 2:
                return
        else:
            stable_rounds = 0
        previous = current
        await page.wait_for_timeout(750)


async def wait_for_tournament_shell(page, timeout_ms: int) -> None:
    """Wait for the RTT Vue app, not for the independently loaded chat widget."""

    await page.wait_for_function(
        r"""() => {
            const text = (document.body?.innerText || '').replace(/\s+/g, ' ').toLocaleLowerCase('ru');
            return text.includes('карточка турнира') && text.includes('рег. номер');
        }""",
        timeout=timeout_ms,
    )


async def wait_for_player_sections(page, route_key: str, timeout_ms: int) -> None:
    headings = _PLAYER_PANEL_HEADINGS.get(route_key, ())
    if not headings:
        return
    await page.wait_for_function(
        r"""headings => {
            const text = (document.body?.innerText || '').replace(/\s+/g, ' ').toLocaleLowerCase('ru');
            const loading = document.querySelector('.content-area .v-progress-circular--indeterminate');
            return !loading && headings.some(heading => text.includes(heading));
        }""",
        arg=list(headings),
        timeout=timeout_ms,
    )


async def expand_player_panels(page, route_key: str, timeout_ms: int) -> list[str]:
    """Open RTT's lazy player accordions and retain every rendered table.

    The public requests/members pages initially render only accordion headings.
    Vuetify mounts the player rows after a heading is clicked and may unmount a
    previous accordion when the next one opens, so each expanded panel is saved
    as a separate HTML fragment.
    """

    wanted_headings = _PLAYER_PANEL_HEADINGS.get(route_key, ())
    if not wanted_headings:
        return []

    fragments: list[str] = []
    headers = page.locator("button.v-expansion-panel-header")
    for wanted in wanted_headings:
        matching_button = None
        expected_rows = 0
        for index in range(await headers.count()):
            button = headers.nth(index)
            text = " ".join((await button.inner_text()).split()).casefold()
            if text.startswith(wanted):
                matching_button = button
                count_match = re.search(r"в списке:\s*(\d+)", text)
                expected_rows = int(count_match.group(1)) if count_match else 0
                break
        if matching_button is None:
            continue

        if await matching_button.get_attribute("aria-expanded") != "true":
            await matching_button.click(timeout=min(timeout_ms, 10_000))
        await wait_until_stable(page, min(timeout_ms, 12_000))
        panel = matching_button.locator(
            "xpath=ancestor::*[contains(concat(' ', normalize-space(@class), ' '), ' v-expansion-panel ')][1]"
        )
        if expected_rows:
            await panel.locator("table tbody tr").nth(expected_rows - 1).wait_for(
                state="attached", timeout=min(timeout_ms, 20_000)
            )
            await wait_until_stable(page, min(timeout_ms, 8_000))
        fragments.append(await panel.evaluate("element => element.outerHTML"))
    return fragments


def append_html_fragments(html: str, fragments: list[str]) -> str:
    if not fragments:
        return html
    payload = "\n<!-- RTT expanded player panels -->\n" + "\n".join(fragments)
    closing_body = html.casefold().rfind("</body>")
    if closing_body < 0:
        return html + payload
    return html[:closing_body] + payload + "\n" + html[closing_body:]


async def fetch_route(context, tour_id: str, route: str, cache_dir: Path, timeout_ms: int) -> dict[str, object]:
    key = route.rstrip("/")
    url = f"{RTT_PUBLIC_ROOT}/{tour_id}/{route}"
    destination = cache_dir / tour_id / f"{key}.html"
    page = await context.new_page()
    api_responses: list[dict[str, object]] = []
    document_responses: list[dict[str, object]] = []
    failed_requests: list[dict[str, str]] = []
    console_errors: list[str] = []
    page_errors: list[str] = []

    def remember_response(response) -> None:
        if "apirtt.mytennis.online" in response.url:
            api_responses.append({"status": response.status, "url": response.url})
        if response.request.resource_type in {"document", "script"}:
            document_responses.append(
                {
                    "status": response.status,
                    "type": response.request.resource_type,
                    "url": response.url,
                }
            )

    def remember_failed_request(request) -> None:
        failure = request.failure
        failed_requests.append(
            {
                "url": request.url,
                "error": str(failure or "request failed"),
                "method": request.method,
                "post_data": (request.post_data or "")[:4000],
            }
        )

    def remember_console(message) -> None:
        if message.type == "error":
            console_errors.append(message.text)

    page.on("response", remember_response)
    page.on("requestfailed", remember_failed_request)
    page.on("console", remember_console)
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        try:
            await page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 12_000))
        except Exception:
            pass
        await wait_until_stable(page, min(timeout_ms, 15_000))
        await wait_for_tournament_shell(page, timeout_ms)
        body_text = await page.locator("body").inner_text()
        body_normalized = " ".join(body_text.split()).casefold()
        if len(body_normalized) < 80 or "турнира undefined" in body_normalized:
            raise RuntimeError("RTT page did not return tournament data")
        await wait_for_player_sections(page, key, timeout_ms)
        player_fragments = await expand_player_panels(page, key, timeout_ms)
        html = append_html_fragments(await page.content(), player_fragments)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".html.tmp")
        temporary.write_text(html, encoding="utf-8")
        temporary.replace(destination)
        return {
            "route": key,
            "ok": True,
            "url": url,
            "path": str(destination),
            "api_responses": api_responses[-20:],
            "document_responses": document_responses[-30:],
        }
    except Exception as exc:
        try:
            failure_title = await page.title()
            failure_body = " ".join((await page.locator("body").inner_text()).split())[:1200]
        except Exception:
            failure_title = ""
            failure_body = ""
        return {
            "route": key,
            "ok": False,
            "url": url,
            "path": str(destination) if destination.exists() else "",
            "error": f"{type(exc).__name__}: {exc}",
            "api_responses": api_responses[-20:],
            "document_responses": document_responses[-30:],
            "failed_requests": failed_requests[-20:],
            "console_errors": console_errors[-10:],
            "page_errors": page_errors[-10:],
            "final_url": page.url,
            "title": failure_title,
            "body_excerpt": failure_body,
        }
    finally:
        await page.close()


def snapshot_result_from_cache(
    tour_id: str,
    args,
    route_results: list[dict[str, object]],
    *,
    cache_warning: str = "",
) -> dict[str, object]:
    pages, paths = read_cached_pages(args.cache_dir, tour_id)
    if not pages:
        error = "Нет свежих или кэшированных страниц турнира."
        if cache_warning:
            error = f"{cache_warning}; {error}"
        return {"tour_id": tour_id, "ok": False, "routes": route_results, "error": error}
    cached_page_times = [
        datetime.fromtimestamp(Path(path).stat().st_mtime).astimezone()
        for path in paths.values()
        if Path(path).exists()
    ]
    snapshot_fetched_at = (
        max(cached_page_times).isoformat(timespec="seconds")
        if cached_page_times
        else datetime.now().astimezone().isoformat(timespec="seconds")
    )
    snapshot = build_snapshot_from_pages(
        tour_id,
        pages,
        fetched_at=snapshot_fetched_at,
        page_paths=paths,
    )
    if not snapshot.title and not snapshot.status and not snapshot.players:
        return {
            "tour_id": tour_id,
            "ok": False,
            "routes": route_results,
            "error": "RTT returned only the application shell without tournament data.",
        }
    failed = [row for row in route_results if not row["ok"]]
    if cache_warning:
        snapshot.warnings.append(f"{cache_warning}; использован последний отдельный кэш.")
    elif failed:
        snapshot.warnings.append(
            "RTT недоступен для части страниц; использован последний отдельный кэш: "
            + ", ".join(str(row["route"]) for row in failed)
        )
    path = save_snapshot(snapshot, args.cache_dir)
    return {
        "tour_id": tour_id,
        "ok": True,
        "snapshot_path": str(path),
        "eligible": snapshot.eligible,
        "players": len(snapshot.players),
        "player_source": snapshot.player_source,
        "routes": route_results,
        "warnings": snapshot.warnings,
    }


async def fetch_tournament(context, semaphore: asyncio.Semaphore, tour_id: str, args) -> dict[str, object]:
    api_error = getattr(args, "api_unavailable_error", "")
    if api_error:
        return snapshot_result_from_cache(
            tour_id,
            args,
            [],
            cache_warning=f"RTT API недоступен: {api_error}",
        )

    async def guarded_route(route: str) -> dict[str, object]:
        async with semaphore:
            return await fetch_route(
                context,
                tour_id,
                route,
                args.cache_dir,
                args.timeout_seconds * 1000,
            )

    selected_routes = (
        tuple(
            next(candidate for candidate in TOURNAMENT_ROUTES if candidate.rstrip("/") == route.rstrip("/"))
            for route in args.route
        )
        if args.route
        else TOURNAMENT_ROUTES
    )
    route_results = await asyncio.gather(*(guarded_route(route) for route in selected_routes))
    return snapshot_result_from_cache(tour_id, args, route_results)


async def async_main(args) -> list[dict[str, object]]:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise SystemExit(
            "Playwright is not available through the project environment. Install project requirements first."
        ) from exc

    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    args.cache_dir = args.cache_dir.resolve()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    browser_path = PROJECT_ROOT / "tmp" / "ms-playwright"
    if browser_path.exists() and not os.environ.get("PLAYWRIGHT_BROWSERS_PATH"):
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(browser_path)

    async with async_playwright() as playwright:
        browser_type = getattr(playwright, args.browser)
        browser = await browser_type.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1600, "height": 2400}, locale="ru-RU")
        try:
            semaphore = asyncio.Semaphore(args.concurrency)
            try:
                response = await context.request.get(
                    RTT_API_HEALTH_URL,
                    headers={
                        "Origin": "https://rtt.mytennis.online",
                        "Referer": "https://rtt.mytennis.online/",
                    },
                    timeout=min(args.timeout_seconds * 1000, 15_000),
                )
                args.api_unavailable_error = "" if response.ok else f"HTTP {response.status}"
            except Exception as exc:
                error_text = str(exc).splitlines()[0]
                args.api_unavailable_error = f"{type(exc).__name__}: {error_text}"
            return await asyncio.gather(
                *(fetch_tournament(context, semaphore, str(tour_id).strip(), args) for tour_id in args.tour_id)
            )
        finally:
            await context.close()
            await browser.close()


def main() -> None:
    ensure_playwright_interpreter()
    args = build_parser().parse_args()
    results = asyncio.run(async_main(args))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps({"ok": all(row.get("ok") for row in results), "results": results}, ensure_ascii=False, indent=2))
    if not all(row.get("ok") for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
