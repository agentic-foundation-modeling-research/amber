from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
import re

from context_scythe.environment.viewport import compute_viewport_bids, filter_axtree_by_bids
from .observation import from_browsergym_dict

logger = logging.getLogger(__name__)


class ReasoningParseError(Exception):
    pass


class ActionParseError(Exception):
    pass


class MemoryParseError(Exception):
    pass


class StateParseError(Exception):
    pass


@dataclass
class Observation:
    """Observation data captured at a single turn of a browser interaction trajectory.

    Stores the raw observation components (accessibility tree, screenshot, tab info)
    from BrowserGym and provides methods to serialize them and compose them into
    chat-formatted message parts suitable for LLM consumption.

    Each observation can include any combination of:
      - A flattened accessibility tree (AXTree) representing the page DOM structure.
      - A screenshot of the current page (not yet supported for message composition).
      - Open browser tab metadata (URLs, titles, active tab index).

    The ``compose_message`` method assembles the enabled components into a list of
    ``{"type": "text", "text": ...}`` content parts that can be embedded in a
    chat message's ``content`` array.

    Attributes:
        axtree: String representation of the page's accessibility tree,
            or None if AXTree observation is disabled.
        screenshot: Raw screenshot data as a list, or None if screenshot
            observation is disabled.
        open_pages_urls: URLs of all open browser tabs, or None if tab info
            is disabled.
        open_pages_titles: Titles of all open browser tabs, or None if tab info
            is disabled.
        active_page_index: Index of the currently active tab, or None if tab
            info is disabled.
        use_axtree: Whether to include the accessibility tree in composed messages.
        use_screenshot: Whether to include the screenshot in composed messages.
        use_tabs_info: Whether to include browser tab information in composed messages.
        axtree_max_tokens: If set, truncates the AXTree string to approximately
            this many tokens (estimated at 4 characters per token) when composing
            messages. None means no truncation.
    """

    axtree: str | None
    viewport_state: dict | None
    extra_element_properties: dict | None
    open_pages_urls: list[str] | None
    open_pages_titles: list[str] | None
    active_page_index: list[int] | None
    last_action_error: str | None
    screenshot: list[Any] | None = None

    use_axtree: bool = True
    filter_viewport: bool = True
    use_screenshot: bool = False
    use_tabs_info: bool = True
    axtree_max_tokens: int | None = None

    def to_json(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "type": "Observation",
            "axtree": self.axtree,
            "viewport_state": self.viewport_state,
            "extra_element_properties": self.extra_element_properties,
            "screenshot": self.screenshot,
            "open_pages_urls": self.open_pages_urls,
            "open_pages_titles": self.open_pages_titles,
            "active_page_index": self.active_page_index,
            "last_action_error": self.last_action_error,
            "use_axtree": self.use_axtree,
            "use_screenshot": self.use_screenshot,
            "use_tabs_info": self.use_tabs_info,
            "axtree_max_tokens": self.axtree_max_tokens,
            "filter_viewport": self.filter_viewport
        }

    @classmethod
    def from_json(cls, data: dict) -> Observation:
        """Deserialize from a JSON-compatible dict."""
        return cls(
            axtree=data["axtree"],
            viewport_state=data.get("viewport_state"),
            extra_element_properties=data.get("extra_element_properties"),
            screenshot=data.get("screenshot", None),
            open_pages_urls=data.get("open_pages_urls"),
            open_pages_titles=data.get("open_pages_titles"),
            active_page_index=data.get("active_page_index"),
            last_action_error=data.get("last_action_error"),
            use_axtree=data.get("use_axtree", True),
            use_screenshot=data.get("use_screenshot", False),
            use_tabs_info=data.get("use_tabs_info", False),
            filter_viewport=data.get("filter_viewport", False),
            axtree_max_tokens=data.get("axtree_max_tokens"),
        )
    
    def viewport_message(self, viewport_state):
        """Return the text representation of the current viewport"""
        msg = "# Current Viewport\n"
        viewport_width = f'Width: {viewport_state["viewport_width"]}'
        viewport_height = f'Hight: {viewport_state["viewport_height"]}'
        device_scale_factor = f'Device Scale Factor: {viewport_state["device_scale_factor"]}'
        scroll_x = f'Scroll X: {viewport_state.get("scroll_x", 0)}'
        scroll_y = f'Scroll Y: {viewport_state.get("scroll_y", 0)}'

        msg = "\n".join([msg, viewport_width, viewport_height, device_scale_factor, scroll_x, scroll_y, "\n"])
        return {"type": "text", "text": msg}
    
    def tabs_message(self, open_pages_urls: list[str], open_pages_titles: list[str], active_page_index: int) -> dict:
        """Return the text representation of the tabs."""
        tabs_text = "# Currently open tabs\n\n"
        for page_index, (url, title) in enumerate(zip(open_pages_urls, open_pages_titles)):
            active = " (active tab)" if page_index == active_page_index else ""
            tabs_text += f"Tab {page_index}{active}\n  Title: {title}\n  URL: {url}\n"
        return {"type": "text", "text": tabs_text}

    def axtree_message(self, axtree) -> dict:
        """Return the text representation of the AXTree."""
        if self.filter_viewport:
            viewport_bids = compute_viewport_bids(self.extra_element_properties)
            axtree = filter_axtree_by_bids(axtree, viewport_bids)

        if self.axtree_max_tokens is not None:
            max_chars = self.axtree_max_tokens  # TODO: Change variable name to max_chars
            if len(axtree) > max_chars:
                axtree = axtree[:max_chars] + "\n... (truncated)"

        axtree_txt = "\n\n# Current page Accessibility Tree\n\n"
        axtree_txt += f"{axtree}\n\n"

        if self.filter_viewport:
            axtree_txt += "The accessibility tree contains the elements from the current viewport.\n\n"

        return {"type": "text", "text": axtree_txt}

    def screenshot_message(self, processed: dict) -> dict:
        """Return the text representation of the screenshot."""
        raise NotImplementedError("Screenshot not supported yet.")
    
    def last_action_error_message(self, last_action_error: str):
        txt = f"# Last action error\n{last_action_error}"
        return {"type": "text", "text": txt}

    def content(self) -> list[dict]:
        """Return the text representation of the trajectory turn."""
        content_parts = []

        if self.use_tabs_info:
            active_tab_index = self.active_page_index[0]
            content_parts.append(
                self.tabs_message(self.open_pages_urls, self.open_pages_titles, active_tab_index)
            )
        if self.use_axtree:
            content_parts.append(self.axtree_message(self.axtree))
        if self.filter_viewport:
            content_parts.append(self.viewport_message(self.viewport_state))
        if self.use_screenshot:
            content_parts.append(self.screenshot_message(self.screenshot))
        if self.last_action_error is not None and self.last_action_error:
            content_parts.append(self.last_action_error_message(self.last_action_error))

        return content_parts


@dataclass
class Response:
    """Model response for a single trajectory turn, containing reasoning and action.

    Stores the complete model output along with its parsed components: the
    chain-of-thought reasoning (extracted from ``<think>...</think>`` tags) and
    the executable action (extracted from ``<action>...</action>`` tags).

    The class provides factory methods for constructing instances from raw model
    output strings, automatically parsing the reasoning and action components.

    Attributes:
        model_full_response: The complete, unmodified text output from the model,
            including both ``<think>`` and ``<action>`` blocks.
        reasoning: The extracted reasoning text from the last ``<think>...</think>``
            block, or None if no think block was found.
        action: The extracted action string from the last ``<action>...</action>``
            block, or None if no action block was found.
    """

    model_full_response: str  # Model output (entire response from the model)
    reasoning: str | None = None # Model reasoning
    action: str | None = None  # Action to be executed by the environment

    # Token-level data from vLLM generation (optional, useful during rollout)
    response_token_ids: list[int] | None = None  # tokens generated by vLLM (after suffix)
    response_log_probs: list[float] | None = None  # per-token log probs

    raise_on_reasoning_parse_error: bool = False # Raise ReasoningParseError when failing to parse reasoning
    raise_on_action_parse_error: bool = False # Raise ActionParseError when failing to parse action

    def to_json(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        data = {
            "type": "Response",
            "model_full_response": self.model_full_response,
            "reasoning": self.reasoning,
            "action": self.action,
        }
        if self.response_token_ids is not None:
            data["response_token_ids"] = self.response_token_ids
        if self.response_log_probs is not None:
            data["response_log_probs"] = self.response_log_probs
        return data

    @classmethod
    def from_json(cls, data: dict) -> Response:
        """Deserialize from a JSON-compatible dict."""
        return cls(
            model_full_response=data["model_full_response"],
            reasoning=data.get("reasoning"),
            action=data.get("action"),
            response_token_ids=data.get("response_token_ids"),
            response_log_probs=data.get("response_log_probs"),
        )

    @staticmethod
    def parse_action(
        model_output: str,
        raise_on_action_parse_error: bool = False,
    ) -> tuple[str, str | None]:
        """Extract text before the last action block and the action.

        Looks for the last <action>...</action> block in the output and returns
        the stripped text before that block plus the stripped action text.
        """
        action_matches = list(
            re.finditer(r"<action>(.*?)</action>", model_output, re.DOTALL)
        )
        if action_matches:
            last_action_match = action_matches[-1]
            action = last_action_match.group(1).strip()
            if not action:
                if raise_on_action_parse_error:
                    raise ActionParseError(
                        f"Could not parse non-empty action from model response {model_output}. "
                        "Ensure that the action is wrapped in <action>...</action> tags and is not empty."
                    )
                return model_output[:last_action_match.start()].strip(), None
            return (
                model_output[:last_action_match.start()].strip(),
                action,
            )

        if raise_on_action_parse_error:
            raise ActionParseError(f"Could not parse action from model response {model_output}. Ensure that the action is wrapped in <action>...</action> tags.")
        else:
            return model_output.strip(), None

    @staticmethod
    def parse_reasoning(
        model_output: str,
        raise_on_reasoning_parse_error: bool = False,
    ) -> str | None:
        """Extract reasoning from reasoning+action (+anything else) output.

        Looks for the last <think>...</think> block in the output.
        """
        think_blocks = re.findall(r"<think>(.*?)</think>", model_output, re.DOTALL)
        if think_blocks:
            return think_blocks[-1].strip()

        # Ideally, we don't want to be here
        if raise_on_reasoning_parse_error:
            raise ReasoningParseError(f"Could not parse reasoning from model response {model_output}. Ensure that the reasoning is wrapped in <think>...</think> tags.")
        else:
            return None

    def content(self) -> list[dict]:
        content_parts = {
            "type": "text",
            "text": self.model_full_response,
        }
        return [content_parts]


@dataclass
class Memory:
    """Model response for a single trajectory turn, containing reasoning and action.

    Stores the complete model output along with its parsed components: the
    chain-of-thought reasoning (extracted from ``<think>...</think>`` tags) and
    the executable action (extracted from ``<action>...</action>`` tags).

    The class provides factory methods for constructing instances from raw model
    output strings, automatically parsing the reasoning and action components.

    Attributes:
        model_full_response: The complete, unmodified text output from the model,
            including both ``<think>`` and ``<action>`` blocks.
        reasoning: The extracted reasoning text from the last ``<think>...</think>``
            block, or None if no think block was found.
        action: The extracted action string from the last ``<action>...</action>``
            block, or None if no action block was found.
    """

    model_full_response: str  # Model output (entire response from the model)
    reasoning: str | None = None # Model reasoning
    memory: str | None = None  # Memory predicted by the model
    state: str | None = None  # State predicted by the model

    raise_on_reasoning_parse_error: bool = False # Raise ReasoningParseError when failing to parse reasoning
    raise_on_memory_parse_error: bool = False # Raise MemoryParseError when failing to parse memory
    raise_on_state_parse_error: bool = False # Raise StateParseError when failing to parse memory

    def to_json(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        data = {
            "type": "Response",
            "model_full_response": self.model_full_response,
            "reasoning": self.reasoning,
            "memory": self.memory,
            "state": self.state,
        }
        return data

    @classmethod
    def from_json(cls, data: dict) -> Response:
        """Deserialize from a JSON-compatible dict."""
        return cls(
            model_full_response=data["model_full_response"],
            reasoning=data.get("reasoning"),
            memory=data.get("memory"),
            state=data.get("state"),
        )

    @staticmethod
    def parse_reasoning(
        model_output: str,
        raise_on_reasoning_parse_error: bool = False,
    ) -> str | None:
        """Extract reasoning from reasoning+action (+anything else) output.

        Looks for the last <think>...</think> block in the output.
        """
        think_blocks = re.findall(r"<think>(.*?)</think>", model_output, re.DOTALL)
        if think_blocks:
            return think_blocks[-1].strip()

        # Ideally, we don't want to be here
        if raise_on_reasoning_parse_error:
            raise ReasoningParseError(f"Could not parse reasoning from model response {model_output}. Ensure that the reasoning is wrapped in <think>...</think> tags.")
        else:
            return None

    @staticmethod
    def parse_memory(
        model_output: str,
        raise_on_memory_parse_error: bool = False,
    ) -> str | None:
        """Extract reasoning from reasoning+action (+anything else) output.

        Looks for the last <memory>...</memory> block in the output.
        """
        memory_blocks = re.findall(r"<memory>(.*?)</memory>", model_output, re.DOTALL)
        if memory_blocks:
            return memory_blocks[-1].strip()

        # Ideally, we don't want to be here
        if raise_on_memory_parse_error:
            raise MemoryParseError(f"Could not parse memory from model response {model_output}. Ensure that the reasoning is wrapped in <memory>...</memory> tags.")
        else:
            return None

    @staticmethod
    def parse_state(
        model_output: str,
        raise_on_state_parse_error: bool = False,
    ) -> str | None:
        """Extract reasoning from reasoning+action (+anything else) output.

        Looks for the last <state>...</state> block in the output.
        """
        memory_blocks = re.findall(r"<state>(.*?)</state>", model_output, re.DOTALL)
        if memory_blocks:
            return memory_blocks[-1].strip()

        # Ideally, we don't want to be here
        if raise_on_state_parse_error:
            raise StateParseError(f"Could not parse state from model response {model_output}. Ensure that the reasoning is wrapped in <state>...</state> tags.")
        else:
            return None

    @classmethod
    def parse_response(cls, model_output: str) -> Memory:
        if not isinstance(model_output, str):
            raise StateParseError("State-memory response must be a string")
        reasoning = Memory.parse_reasoning(model_output, raise_on_reasoning_parse_error=True)
        memory = Memory.parse_memory(model_output, raise_on_memory_parse_error=True)
        state = Memory.parse_state(model_output, raise_on_state_parse_error=True)
        return cls(
            model_full_response=model_output,
            reasoning=reasoning,
            memory=memory,
            state=state
        )

    def content(self) -> list[dict]:
        content_parts = {
            "type": "text",
            "text": self.model_full_response,
        }
        return [content_parts]


@dataclass
class StepData:
    step_num: int
    observation: Observation
    response: Response = None

    def to_json(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        data = {
            "type": "StepData",
            "step_num": self.step_num,
            "observation": self.observation.to_json(),
        }
        if self.response is not None:
            data["response"] = self.response.to_json()
        return data

    @classmethod
    def from_json(cls, data: dict) -> StepData:
        """Deserialize from a JSON-compatible dict."""
        return cls(
            step_num=data["step_num"],
            observation=Observation.from_json(data["observation"]),
            response=(
                Response.from_json(data["response"])
                if data.get("response") is not None
                else None
            ),
        )

    def observation_message_content(self) -> list[dict]:
        return self.observation.content()
    
    def response_message_content(self) -> list[dict]:
        return self.response.content()
    

@dataclass
class StepDataWithMemory(StepData):
    memory: str = None

    def to_json(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        data = super().to_json()
        data["type"] = "StepDataWithMemory"
        data["memory"] = self.memory
        return data

    @classmethod
    def from_json(cls, data: dict) -> StepDataWithMemory:
        """Deserialize from a JSON-compatible dict."""
        step_data = super().from_json(data)
        return cls(
            step_num=step_data.step_num,
            observation=step_data.observation,
            response=step_data.response,
            memory=data.get("memory"),
        )


@dataclass
class StepDataWithStateMemory(StepData):
    memory: Memory = None

    def to_json(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        data = super().to_json()
        data["type"] = "StepDataWithStateMemory"
        data["memory"] = self.memory.to_json()
        return data

    @classmethod
    def from_json(cls, data: dict) -> StepDataWithMemory:
        """Deserialize from a JSON-compatible dict."""
        step_data = super().from_json(data)
        return cls(
            step_num=step_data.step_num,
            observation=step_data.observation,
            response=step_data.response,
            memory=Memory.from_json(data["memory"])
            if data.get("memory") is not None
            else None,
        )

    def memory_message_content(self) -> list[dict]:
        return self.memory.content()


@dataclass
class TrajectoryData:
    goal: str
    calculator_url: str
    site_urls: dict[str, str]
    reward: float = None
    terminated: bool = None
    truncated: bool = None
    steps: list[StepData] = field(default_factory=list)

    def add_step(self, step_data: StepData):
        assert step_data.step_num == len(self.steps)
        self.steps.append(step_data)

    def update_reward(self, reward):
        self.reward = reward

    def to_json(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "type": "TrajectoryData",
            "goal": self.goal,
            "calculator_url": self.calculator_url,
            "site_urls": self.site_urls,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "steps": [step.to_json() for step in self.steps],
        }

    @classmethod
    def from_json(cls, data: dict) -> TrajectoryData:
        """Deserialize from a JSON-compatible dict."""
        return cls(
            goal=data["goal"],
            calculator_url=data["calculator_url"],
            site_urls=data["site_urls"],
            reward=data.get("reward"),
            terminated=data.get("terminated"),
            truncated=data.get("truncated"),
            steps=[StepData.from_json(step) for step in data.get("steps", [])],
        )

    def __len__(self):
        return len(self.steps)


@dataclass
class TrajectoryDataWithMemory(TrajectoryData):
    steps: list[StepDataWithMemory] = field(default_factory=list)

    def to_json(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        data = super().to_json()
        data["type"] = "TrajectoryDataWithMemory"
        return data

    @classmethod
    def from_json(cls, data: dict) -> TrajectoryDataWithMemory:
        """Deserialize from a JSON-compatible dict."""
        return cls(
            goal=data["goal"],
            calculator_url=data["calculator_url"],
            site_urls=data["site_urls"],
            reward=data.get("reward"),
            terminated=data.get("terminated"),
            truncated=data.get("truncated"),
            steps=[
                StepDataWithMemory.from_json(step)
                for step in data.get("steps", [])
            ],
        )


@dataclass
class TrajectoryDataWithStateMemory(TrajectoryDataWithMemory):
    steps: list[StepDataWithStateMemory] = field(default_factory=list)

    def to_json(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        data = super().to_json()
        data["type"] = "TrajectoryDataWithStateMemory"
        return data

    @classmethod
    def from_json(cls, data: dict) -> TrajectoryDataWithStateMemory:
        """Deserialize from a JSON-compatible dict."""
        return cls(
            goal=data["goal"],
            calculator_url=data["calculator_url"],
            site_urls=data["site_urls"],
            reward=data.get("reward"),
            terminated=data.get("terminated"),
            truncated=data.get("truncated"),
            steps=[
                StepDataWithStateMemory.from_json(step)
                for step in data.get("steps", [])
            ],
        )
