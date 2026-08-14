"""Observation compaction helpers for rollout server responses."""

from __future__ import annotations

import base64
import io
from typing import Any

from browsergym.utils.obs import flatten_axtree_to_str

from context_scythe.env_server.schemas import CompactObservation


def from_browsergym_dict(
    obs: dict,
    use_axtree=True,
    use_screenshot=False,
):
    """
    Create the observation class from the BrowserGym dict
    """
    obs_dict = dict()
    extra_properties = obs.get("extra_element_properties")
    obs_dict["extra_element_properties"] = extra_properties

    if use_axtree:
        axtree_object = obs.get("axtree_object")
    
        axtree = flatten_axtree_to_str(
            axtree_object,
            extra_properties=extra_properties,
            with_visible=True,
            with_clickable=True,
        )
        viewport_state = obs.get("viewport_state")

        obs_dict["axtree"] = axtree
        obs_dict["viewport_state"] = viewport_state

    active_page_index = obs.get("active_page_index", None)
    # Force cast to int, as it can be a np.array
    active_page_index = [int(i) for i in active_page_index]
    
    open_pages_titles = obs.get("open_pages_titles", None)
    open_pages_urls = obs.get("open_pages_urls", None)

    assert active_page_index is not None
    assert open_pages_titles is not None
    assert open_pages_urls is not None

    obs_dict["active_page_index"] = active_page_index
    obs_dict["open_pages_titles"] = open_pages_titles
    obs_dict["open_pages_urls"] = open_pages_urls

    if use_screenshot:
        obs_dict["screenshot"] = _encode_screenshot_as_png_base64(obs.get("screenshot"))
    
    last_action_error = obs.get("last_action_error")
    obs_dict["last_action_error"] = last_action_error

    return obs_dict


def compact_observation(
    obs: dict[str, Any],
    use_axtree: bool = True,
    use_screenshots: bool = False,
) -> CompactObservation:
    """Return the small JSON-safe observation payload used by the rollout API."""
    observation_dict = from_browsergym_dict(
        obs,
        use_axtree=use_axtree,
        use_screenshot=use_screenshots,
    )

    observation = CompactObservation(
        axtree=observation_dict.get("axtree"),
        viewport_state=observation_dict.get("viewport_state"),
        extra_element_properties=observation_dict.get("extra_element_properties"),
        active_page_index=_coerce_active_page_index(observation_dict.get("active_page_index", 0)),
        open_pages_titles=_coerce_tuple_list(observation_dict.get("open_pages_titles")),
        open_pages_urls=_coerce_tuple_list(observation_dict.get("open_pages_urls")),
        last_action_error=observation_dict.get("last_action_error"),
        screenshot=observation_dict.get("screenshot"),
    )

    return observation


def _coerce_active_page_index(value: Any) -> list[int]:
    if isinstance(value, tuple):
        value = list(value)
    elif isinstance(value, int):
        value = [value]
    
    return value


def _coerce_tuple_list(value: Any) -> list[str]:
    if isinstance(value, tuple):
        return list(value)
    return value


def _encode_screenshot_as_png_base64(screenshot: Any) -> str | None:
    if screenshot is None:
        return None
    if isinstance(screenshot, str):
        return screenshot
    if isinstance(screenshot, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(screenshot)).decode("ascii")

    from PIL import Image
    import numpy as np

    if isinstance(screenshot, Image.Image):
        image = screenshot
    else:
        array = np.asarray(screenshot)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(array)

    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    with io.BytesIO() as output:
        image.save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")
