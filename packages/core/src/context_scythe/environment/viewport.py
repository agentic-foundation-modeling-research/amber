"""Viewport-based filtering for the BrowserGym accessibility tree.

The full flattened AXTree can be very large on content-heavy pages. When the
agent only needs to reason about what is currently visible, we can keep just
the bids whose bounding boxes intersect the current scroll viewport, along
with their ``StaticText`` children and ancestors for tree structure.

Ported with light trimming from the reference implementation in
``efficient-context/efficient_context/elements.py`` and ``artifacts.py``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal, Optional, Set

logger = logging.getLogger(__name__)


def compute_viewport_bids(
    extra_element_properties: dict
) -> Set[str]:

    bids: Set[str] = set()
    for bid, props in extra_element_properties.items():
        visibility = props.get("visibility")
        if visibility is not None and visibility >= 0.5:
            bids.add(str(bid))

    return bids


_BID_LINE_PATTERN = re.compile(r"\[(\d+)\]")


def filter_axtree_by_bids(axtree_str: str, viewport_bids: Set[str]) -> str:
    """Post-filter a flattened AXTree string to viewport bids and their context.

    Keeps:
      1. Lines whose bid is in ``viewport_bids``.
      2. Immediate ``StaticText`` children (depth+1) of kept lines.
      3. All ancestors of kept lines (to preserve tree structure).
    """
    lines = axtree_str.split("\n")

    line_depths = []
    line_bids = []
    for line in lines:
        stripped = line.lstrip("\t")
        depth = len(line) - len(stripped)
        line_depths.append(depth)

        match = _BID_LINE_PATTERN.search(line)
        line_bids.append(match.group(1) if match else None)

    keep = [False] * len(lines)

    for i, bid in enumerate(line_bids):
        if bid is not None and bid in viewport_bids:
            keep[i] = True
            for j in range(i + 1, len(lines)):
                if line_depths[j] <= line_depths[i]:
                    break
                if line_depths[j] == line_depths[i] + 1 and "StaticText" in lines[j]:
                    keep[j] = True

    for i in range(len(lines)):
        if keep[i]:
            target_depth = line_depths[i] - 1
            for j in range(i - 1, -1, -1):
                if target_depth < 0:
                    break
                if line_depths[j] == target_depth:
                    keep[j] = True
                    target_depth -= 1

    return "\n".join(line for i, line in enumerate(lines) if keep[i])


def get_viewport_state(page: Any) -> Optional[dict]:
    """Query a Playwright ``Page`` for its current CSS viewport/scroll/DPR state.

    Returns ``None`` if any attribute cannot be read (e.g. the page has been
    closed). Callers treat ``None`` as "no viewport info available" and fall
    back to the full axtree. ``window.innerWidth`` and ``window.innerHeight``
    are used because filtering must match the live CSS viewport, regardless of
    how the browser was launched.
    """
    try:
        viewport = page.viewport_size
        if viewport is None:
            return None
        viewport_width = page.evaluate("window.innerWidth")
        viewport_height = page.evaluate("window.innerHeight")
        scale_candidates = [1.0]
        browsergym_scale_factor = getattr(page, "_bgym_scale_factor", None)
        if browsergym_scale_factor:
            scale_candidates.append(browsergym_scale_factor)
        if viewport_width:
            scale_candidates.append(viewport["width"] / viewport_width)
        if viewport_height:
            scale_candidates.append(viewport["height"] / viewport_height)
        bbox_scale_factor = max(scale_candidates)
        return {
            "viewport_width": viewport_width,
            "viewport_height": viewport_height,
            "device_scale_factor": page.evaluate("window.devicePixelRatio"),
            "bbox_coordinate_space": "viewport",
            "bbox_scale_factor": bbox_scale_factor,
            "scroll_x": page.evaluate("window.scrollX"),
            "scroll_y": page.evaluate("window.scrollY"),
        }
    except Exception as exc:
        logger.debug("Failed to read viewport state from page: %s", exc)
        return None
