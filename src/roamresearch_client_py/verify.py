from __future__ import annotations

from .diff import DiffResult, diff_block_trees, parse_existing_blocks
from .gfm_to_roam import gfm_to_blocks


def diff_page_against_markdown(page: dict, markdown: str) -> DiffResult:
    """
    Compute a verification diff between a fetched Roam page dict and desired markdown.

    Intended usage: after applying batch-actions, fetch the page again and confirm
    that the diff is empty. If it isn't, treat as a warning signal for partial
    application or unexpected Roam behavior.
    """
    page_uid = page.get(":block/uid")
    if not page_uid:
        raise ValueError("page missing :block/uid")

    existing_blocks = parse_existing_blocks(page)
    desired_blocks = gfm_to_blocks(markdown, page_uid)
    return diff_block_trees(existing_blocks, desired_blocks, page_uid)

