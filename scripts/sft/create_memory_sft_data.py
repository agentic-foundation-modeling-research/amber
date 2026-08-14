import json
import os
from pathlib import Path
import argparse

from datasets import Dataset, Features, List, Value
from browsergym.core.action.highlevel import HighLevelActionSet
from context_scythe.agents import TrajectoryDataWithMemory, SingleTurnWithMemoryPromptBuilder

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def parse_args():
    parser = argparse.ArgumentParser(description="Create a Hugging Face SFT dataset from saved trajectories.")
    parser.add_argument(
        "--trajectories_dir",
        required=True,
        type=Path,
        help="Directory containing trajectory JSON files.",
    )
    parser.add_argument(
        "--memories_dir",
        required=True,
        type=Path,
        help="Directory containing per-trajectory memory JSONL files.",
    )
    parser.add_argument(
        "--dataset_save_dir",
        required=True,
        type=Path,
        help="Directory where the Hugging Face dataset will be materialized.",
    )
    parser.add_argument(
        "--memory_format",
        default="append",
        type=str,
        help="Directory where the Hugging Face dataset will be materialized.",
    )
    parser.add_argument(
        "--cache_dir",
        default=DATA_DIR / "cache",
        type=Path,
        help="Directory where Hugging Face Datasets should cache generated Arrow files.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the created dataset to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub-path",
        default=None,
        help="Hugging Face Hub dataset repo path, for example 'username/dataset-name'. Required with --push-to-hub.",
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Create or update the Hub dataset repo as private.",
    )
    parser.add_argument(
        "--hub-split",
        default="train",
        help="Dataset split name to use on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub-commit-message",
        default="Upload SFT dataset",
        help="Commit message to use when pushing the dataset to the Hub.",
    )
    return parser.parse_args()


def compose_trajectory_messages(trajectory_path: Path, memories_file: Path, memory_format: str):

    traj_id = trajectory_path.name.split(".")[0]

    with open(trajectory_path, "r") as f:
        trajectory_dict = json.load(f)

    with open(memories_file, "r") as f:
        memories = {}
        for line in f:
            step_memory = json.loads(line)
            # Artificially insert think tags as it might confuse the model
            # if one turn as thinking and next does not
            memories[step_memory["step"]] = "<think>\n" + step_memory["memory"] + "\n</think>"

    trajectory_data: TrajectoryDataWithMemory = TrajectoryDataWithMemory.from_json(trajectory_dict)
    for step_data in trajectory_data.steps:
        step_data.memory = memories[step_data.step_num]

    action_set = HighLevelActionSet(["webarena"])

    trajectory_messages = []

    prompt_builder = SingleTurnWithMemoryPromptBuilder(memory_format=memory_format)
    for step_data in trajectory_data.steps:
        step_num = step_data.step_num
        messages = prompt_builder.build_messages(
            step_num=step_num,
            mode="compression",
            trajectory_data=trajectory_data,
            action_set=action_set,
            current_step_data=step_data
        )
        sft_messages = messages["prompt"] + [messages["compression_response"]]
        sft_messages = prompt_builder.flatten_messages(sft_messages)
        trajectory_messages.append({"traj_id": traj_id, "step_num": step_num, "prompt": sft_messages})

    return trajectory_messages


SFT_FEATURES = Features({
    "traj_id": Value("string"),
    "step_num": Value("int32"),
    "prompt": List({
        "role": Value("string"),
        "content": Value("string"),
    }),
})


def iter_all_trajectory_messages(trajectories_dir: Path, memories_dir: Path, memory_format: str):
    for memories_file in sorted(memories_dir.glob("*.jsonl")):
        trajectory_path = trajectories_dir / f"{memories_file.stem}.json"
        if not trajectory_path.exists():
            continue
        yield from compose_trajectory_messages(trajectory_path, memories_file, memory_format=memory_format)


def create_hf_dataset(trajectories_dir: Path, memories_dir: Path, dataset_save_dir: Path, cache_dir: Path, memory_format: str):
    dataset = Dataset.from_generator(
        lambda: iter_all_trajectory_messages(trajectories_dir, memories_dir, memory_format),
        features=SFT_FEATURES,
        cache_dir=cache_dir,
    )
    dataset.save_to_disk(dataset_save_dir)
    return dataset


def push_dataset_to_hub(dataset: Dataset, hub_path: str, private: bool, split: str, commit_message: str):
    token = os.environ.get("HF_TOKEN")
    if token is None:
        raise ValueError("HF_TOKEN must be set in the environment when using --push-to-hub.")

    dataset.push_to_hub(
        hub_path,
        private=private,
        split=split,
        token=token,
        commit_message=commit_message,
    )


def main(args):
    dataset = create_hf_dataset(
        trajectories_dir=Path(args.trajectories_dir),
        memories_dir=Path(args.memories_dir),
        dataset_save_dir=Path(args.dataset_save_dir),
        cache_dir=Path(args.cache_dir),
        memory_format=args.memory_format
    )

    if args.push_to_hub:
        if args.hub_path is None:
            raise ValueError("--hub-path is required when using --push-to-hub.")
        push_dataset_to_hub(
            dataset=dataset,
            hub_path=args.hub_path,
            private=args.hub_private,
            split=args.hub_split,
            commit_message=args.hub_commit_message,
        )


if __name__ == "__main__":
    main(parse_args())
