import asyncio
from typing import TYPE_CHECKING

from pyppeteer import launch

if TYPE_CHECKING:
    from pathlib import Path


async def take_screenshot(
    urls: dict[int, list[str]],
    output_dir: "Path",
    ids_to_use: list[int] | None = None,
    wait_for: int = 60 * 1000,
):
    output_screenshot_dir = output_dir / "screenshots"
    output_screenshot_dir.mkdir(parents=True, exist_ok=True)
    if ids_to_use is None:
        ids_to_use = list(urls.keys())
    for id in ids_to_use:
        for i, url in enumerate(urls[id]):
            print(f"Processing url {i} for id {id}, waiting for {wait_for / 1000}s before screenshot")
            browser = await launch()
            page = await browser.newPage()
            await page.setViewport(
                {
                    "width": 1920,
                    "height": 1080,
                },
            )
            await page.goto(
                url,
                {"waitUntil": "networkidle0", "timeout": 0},
            )
            await page.waitFor(wait_for)
            await page.screenshot(
                {"path": str(output_screenshot_dir / f"{id}-screenshot-{i}.png")},
            )
            await browser.close()


def run_screenshot_loop(
    urls: dict[int, list[str]],
    output_dir: "Path",
    ids_to_use: list[int] | None = None,
    wait_for: int = 60 * 1000,
):
    asyncio.get_event_loop().run_until_complete(
        take_screenshot(urls=urls, output_dir=output_dir, ids_to_use=ids_to_use, wait_for=wait_for),
    )
