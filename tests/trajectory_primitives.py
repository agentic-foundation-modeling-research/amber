"""Shared test primitives for trajectory data tests."""

from browsergym.core.action.highlevel import HighLevelActionSet

from context_scythe.agents.trajectory_data import (
    Observation,
    Response,
    StepData,
    TrajectoryData,
    StepDataWithMemory,
    TrajectoryDataWithMemory,
)

DUMMY_AXTREE = (
    "[1] RootWebArea 'Example'\n"
    "  [2] heading 'Welcome'\n"
    "  [3] link 'More info'\n"
    "  [4] button 'Submit'"
)

DUMMY_RESPONSE_TEXT = (
    '<think>\nI see the page. I will click the link.\n</think>\n'
    '<action>\nclick("3")\n</action>'
)
DUMMY_LAST_ACTION_ERROR = "This is a non-empty error."

DUMMY_SUMMARY_TEXT = "Visited Example page. Clicked 'More info' link (bid 3)."

DUMMY_OPEN_PAGES_TITLES = ("Title 0", "Title 1")
DUMMY_OPEN_PAGES_URLS = ("http://dummy0", "http://dummy1")
ACTIVE_PAGE_INDEX = [0]
DUMMY_VIEWPORT_STATE = {
    "viewport_width": 1280,
    "viewport_height": 720,
    "device_scale_factor": 1,
    "bbox_coordinate_space": "viewport",
    "bbox_scale_factor": 1,
    "scroll_x": 0,
    "scroll_y": 0,
}
DUMMY_EXTRA_ELEMENT_PROPERTIES = {
    str(bid): {
        "visibility": 1.0,
        "bbox": [0, bid * 20, 100, 20],
        "clickable": False,
        "set_of_marks": False,
    }
    for bid in range(1, 5)
}


def make_action_set() -> HighLevelActionSet:
    return HighLevelActionSet(
        subsets=["webarena"],
        strict=False,
        multiaction=False,
        demo_mode="off",
    )


def make_observation(
    axtree_text: str = DUMMY_AXTREE,
    last_action_error="",
    use_axtree: bool = True,
    use_screenshot: bool = False,
    use_tabs_info: bool = True,
    filter_viewport: bool = False,
    axtree_max_tokens: int | None = None,
    open_pages_urls: list[str] | None = DUMMY_OPEN_PAGES_URLS,
    open_pages_titles: list[str] | None = DUMMY_OPEN_PAGES_TITLES,
    active_page_index: int | None = ACTIVE_PAGE_INDEX,
    viewport_state: dict | None = DUMMY_VIEWPORT_STATE,
    extra_element_properties: dict | None = DUMMY_EXTRA_ELEMENT_PROPERTIES,
) -> Observation:
    return Observation(
        axtree=axtree_text,
        viewport_state=viewport_state,
        extra_element_properties=extra_element_properties,
        screenshot=None,
        open_pages_urls=open_pages_urls,
        open_pages_titles=open_pages_titles,
        active_page_index=active_page_index,
        last_action_error=last_action_error,
        use_axtree=use_axtree,
        use_screenshot=use_screenshot,
        use_tabs_info=use_tabs_info,
        filter_viewport=filter_viewport,
        axtree_max_tokens=axtree_max_tokens,
    )


def make_response(text: str = DUMMY_RESPONSE_TEXT) -> Response:
    reasoning = Response.parse_reasoning(text)
    _, action = Response.parse_action(text)
    return Response(
        model_full_response=text,
        reasoning=reasoning,
        action=action,
    )


def build_two_step_single_turn(goal="Find laptop price") -> TrajectoryData:
    """Build a 2-turn CompressedTrajectory with all fields populated."""
    make_action_set()
    trajectory = TrajectoryData(
        goal=goal,
        calculator_url="http://dummy.html",
        site_urls={"dummy": "http://dummy_site.com"}
    )

    observation = make_observation()
    response = make_response()
    step_data = StepData(0, observation, response)
    trajectory.add_step(step_data)

    observation = make_observation(axtree_text="[5] RootWebArea 'Shop'\n  [6] heading 'Laptops'")
    response = make_response(
        '<think>\nI see laptops. Let me click 15 inch.\n</think>\n'
        '<action>\nclick("7")\n</action>'
    )
    step_data = StepData(1, observation, response)
    trajectory.add_step(step_data)

    return trajectory


def build_two_step_single_turn_w_mem(goal="Find laptop price") -> TrajectoryData:
    """Build a 2-turn CompressedTrajectory with all fields populated."""
    make_action_set()
    trajectory = TrajectoryDataWithMemory(
        goal=goal,
        calculator_url="http://dummy.html",
        site_urls={"dummy": "http://dummy_site.com"}
    )

    observation = make_observation()
    response = make_response()
    memory = "Step 0 Memory."
    step_data = StepDataWithMemory(0, observation, response, memory=memory)
    trajectory.add_step(step_data)

    observation = make_observation(axtree_text="[5] RootWebArea 'Shop'\n  [6] heading 'Laptops'")
    response = make_response(
        '<think>\nI see laptops. Let me click 15 inch.\n</think>\n'
        '<action>\nclick("7")\n</action>'
    )
    memory = "Step 1 Memory."
    step_data = StepDataWithMemory(1, observation, response, memory=memory)
    trajectory.add_step(step_data)

    return trajectory
