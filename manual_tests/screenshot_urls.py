import asyncio
from typing import TYPE_CHECKING

from pyppeteer import launch

if TYPE_CHECKING:
    from pathlib import Path


async def take_screenshot(
    urls: dict[int, list[str]],
    output_dir: "Path",
    ids_to_use: list[int] | None = None,
):
    output_screenshot_dir = output_dir / "screenshots"
    output_screenshot_dir.mkdir(parents=True, exist_ok=True)
    if ids_to_use is None:
        ids_to_use = list(urls.keys())
    for id in ids_to_use:
        for i, url in enumerate(urls[id]):
            print(f"Processing url {i} for id {id}")
            browser = await launch()
            page = await browser.newPage()
            await page.setViewport(
                {
                    "width": 1920,
                    "height": 1080,
                }
            )
            await page.goto(
                url,
                {
                    "waitUntil": "networkidle0",
                },
            )
            await page.screenshot(
                {"path": str(output_screenshot_dir / f"{id}-screenshot-{i}.png")},
            )
            await browser.close()


def run_screenshot_loop(urls: dict[int, list[str]], output_dir: "Path", ids_to_use: list[int] | None = None):
    asyncio.get_event_loop().run_until_complete(
        take_screenshot(
            urls=urls,
            output_dir=output_dir,
            ids_to_use=ids_to_use,
        )
    )
