import json
import os
from pathlib import Path
import argparse

from datasets import Dataset, Features, List, Value
from browsergym.core.action.highlevel import HighLevelActionSet
from context_scythe.agents import TrajectoryData, SingleTurnPromptBuilder

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
        "--dataset_save_dir",
        required=True,
        type=Path,
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


def compose_trajectory_messages(trajectory_path: Path):

    traj_id = trajectory_path.name.split(".")[0]

    with open(trajectory_path, "r") as f:
        trajectory_dict = json.load(f)

    trajectory_data: TrajectoryData = TrajectoryData.from_json(trajectory_dict)

    action_set = HighLevelActionSet(["webarena"])

    trajectory_messages = []

    prompt_builder = SingleTurnPromptBuilder()
    for step_data in trajectory_data.steps:
        step_num = step_data.step_num
        messages = prompt_builder.build_messages(
            step_num=step_num,
            trajectory_data=trajectory_data,
            action_set=action_set,
            current_step_data=step_data
        )
        sft_messages = messages["prompt"] + messages["response"]
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


def iter_all_trajectory_messages(trajectories_dir: Path):
    for trajectory_path in sorted(trajectories_dir.glob("*.json")):
        yield from compose_trajectory_messages(trajectory_path)


def create_hf_dataset(trajectories_dir: Path, dataset_save_dir: Path, cache_dir: Path):
    dataset = Dataset.from_generator(
        lambda: iter_all_trajectory_messages(trajectories_dir),
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
        num_shards=1
    )


def main(args):
    dataset = create_hf_dataset(
        trajectories_dir=Path(args.trajectories_dir),
        dataset_save_dir=Path(args.dataset_save_dir),
        cache_dir=Path(args.cache_dir),
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
