"""Create compression SFT data from raw state-memory JSONL labels."""

import argparse
import copy
import json
import os
from pathlib import Path

from browsergym.core.action.highlevel import HighLevelActionSet
from datasets import Dataset, Features, List, Value

from context_scythe.agents import (
    SingleTurnWithStateMemoryPromptBuilder,
    TrajectoryDataWithStateMemory,
)
from context_scythe.datagen.state_memory_sft import load_state_memory_jsonl


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create state-memory compression SFT data.")
    parser.add_argument("--trajectories_dir", required=True, type=Path)
    parser.add_argument("--memories_dir", required=True, type=Path)
    parser.add_argument("--dataset_save_dir", required=True, type=Path)
    parser.add_argument("--cache_dir", default=DATA_DIR / "cache", type=Path)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-path", default=None)
    parser.add_argument("--hub-private", action="store_true")
    parser.add_argument("--hub-split", default="train")
    parser.add_argument("--hub-commit-message", default="Upload state-memory SFT dataset")
    return parser.parse_args(argv)


def compose_trajectory_messages(trajectory_path: Path, memories_file: Path) -> list[dict]:
    with trajectory_path.open() as handle:
        source = json.load(handle)

    # Source trajectories can come from any trajectory class. Synthetic labels replace
    # any existing memory so deserialization always uses the state-memory representation.
    trajectory_dict = copy.deepcopy(source)
    for step in trajectory_dict.get("steps", []):
        step["memory"] = None

    trajectory = TrajectoryDataWithStateMemory.from_json(trajectory_dict)
    expected_steps = [step.step_num for step in trajectory.steps]
    labels = load_state_memory_jsonl(memories_file, expected_steps=expected_steps)
    for step in trajectory.steps:
        step.memory = labels[step.step_num]

    action_set = HighLevelActionSet(["webarena"])
    prompt_builder = SingleTurnWithStateMemoryPromptBuilder()
    examples = []
    for step in trajectory.steps:
        messages = prompt_builder.build_messages(
            step_num=step.step_num,
            mode="compression",
            trajectory_data=trajectory,
            action_set=action_set,
            current_step_data=step,
        )
        sft_messages = messages["prompt"] + [messages["compression_response"]]
        examples.append(
            {
                "traj_id": trajectory_path.stem,
                "step_num": step.step_num,
                "prompt": prompt_builder.flatten_messages(sft_messages),
            }
        )
    return examples


SFT_FEATURES = Features(
    {
        "traj_id": Value("string"),
        "step_num": Value("int32"),
        "prompt": List({"role": Value("string"), "content": Value("string")}),
    }
)


def iter_all_trajectory_messages(trajectories_dir: Path, memories_dir: Path):
    for memories_file in sorted(memories_dir.glob("*.jsonl")):
        trajectory_path = trajectories_dir / f"{memories_file.stem}.json"
        if trajectory_path.exists():
            yield from compose_trajectory_messages(trajectory_path, memories_file)


def create_hf_dataset(
    trajectories_dir: Path,
    memories_dir: Path,
    dataset_save_dir: Path,
    cache_dir: Path,
):
    dataset = Dataset.from_generator(
        lambda: iter_all_trajectory_messages(trajectories_dir, memories_dir),
        features=SFT_FEATURES,
        cache_dir=cache_dir,
    )
    dataset.save_to_disk(dataset_save_dir)
    return dataset


def push_dataset_to_hub(dataset, path: str, private: bool, split: str, message: str):
    token = os.environ.get("HF_TOKEN")
    if token is None:
        raise ValueError("HF_TOKEN must be set when using --push-to-hub.")
    dataset.push_to_hub(
        path,
        private=private,
        split=split,
        token=token,
        commit_message=message,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = create_hf_dataset(
        trajectories_dir=args.trajectories_dir,
        memories_dir=args.memories_dir,
        dataset_save_dir=args.dataset_save_dir,
        cache_dir=args.cache_dir,
    )
    if args.push_to_hub:
        if args.hub_path is None:
            raise ValueError("--hub-path is required with --push-to-hub.")
        push_dataset_to_hub(
            dataset,
            path=args.hub_path,
            private=args.hub_private,
            split=args.hub_split,
            message=args.hub_commit_message,
        )


if __name__ == "__main__":
    main()
