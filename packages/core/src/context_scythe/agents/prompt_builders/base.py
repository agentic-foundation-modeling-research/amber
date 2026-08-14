from typing import Any
from ..trajectory_data import TrajectoryData


class BasePromptBuilder:

    def __init__(self, use_privileged_information=False):
        self.use_privileged_information = use_privileged_information
    
    def build_messages(self, trajectory_data: TrajectoryData) -> list[dict]:
        raise NotImplementedError
    
    def flatten_content(self, content: Any):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return str(content)

    def flatten_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        flattened = []
        for message in messages:
            flattened.append({
                **message,
                "content": self.flatten_content(message.get("content", "")),
            })
        return flattened