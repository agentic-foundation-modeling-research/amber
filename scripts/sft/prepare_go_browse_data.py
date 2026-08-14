import argparse
import os
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

from context_scythe.agents import Observation, Response, StepDataWithMemory, TrajectoryDataWithMemory
from context_scythe.environment.const import SITE_URL_TEMPLATES

# The host URL used by Go-Browse. Reverse engineered from GoBrowse data
GOBROWSE_HOST_URL = "http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com"
# Ports have been reverse engineered
PORTS = {
    "map": 3000,
    "shopping": 7770,
    "shopping_admin": 7780,
    "reddit": 9999,
    "homepage": 7564, # not used by go-browse but we need it for our prompts
    "calculator": 7564 # not used by go-browse but we need it for our prompts
}

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
WRITE_TO_JSONL = False
RAW_DATA_PATH = DATA_DIR / "raw_data.jsonl"

TRAJ_METADATA_DIR = DATA_DIR / "sft_data"
TRAJ_SAVE_DIR = TRAJ_METADATA_DIR / "trajectories"
METADATA_FILE = TRAJ_METADATA_DIR / "metadata.jsonl"

total_lines = 196_462 # Hardcode for now. TODO: maybe make this more flexible?


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DATA_DIR,
        help="Directory used for cached and prepared data.",
    )
    parser.add_argument(
        "--traj_save_dir",
        type=Path,
        default=TRAJ_SAVE_DIR,
        help="Directory where trajectory JSON files are written.",
    )
    parser.add_argument(
        "--raw_data_path",
        type=Path,
        default=RAW_DATA_PATH,
        help="Path to the raw Go-Browse JSONL data.",
    )
    parser.add_argument(
        "--metadata_file_path",
        type=Path,
        default=METADATA_FILE,
        help="Path where selected trajectory metadata JSONL is written.",
    )
    parser.add_argument(
        "--write_to_jsonl",
        action="store_true",
        default=WRITE_TO_JSONL,
        help="Download the Hugging Face dataset and write it to raw_data_path before processing.",
    )
    parser.add_argument(
        "--num_duplicate_traj",
        type=int,
        default=1,
        help="Number of shortest trajectories to keep per intent. Use -1 to keep all.",
    )
    args = parser.parse_args()
    if args.num_duplicate_traj < -1:
        parser.error("--num_duplicate_traj must be -1 or greater.")
    return args


def should_skip_trajectory(traj_data, domain, steps):
    if "gitlab" in str(domain).lower():
        return True
    if not traj_data["success"] or len(steps) < 1:
        return True
    # TODO: is ignoring infeasible wise?
    # if steps[-1].response.action and "report_infeasible" in steps[-1].response.action and traj_data["reward"] > 0:
    #     return True
    return False


def format_url(host, site):
    port = PORTS[site]
    url = SITE_URL_TEMPLATES[site].format(host=host, port=port)
    return url


def save_trajectory(traj_key, traj_data, domain, goal, steps, traj_save_dir):
    # Create the sites map
    calculator_url = format_url(GOBROWSE_HOST_URL, "calculator")
    site_urls = {domain: format_url(GOBROWSE_HOST_URL, domain)}

    trajectory = TrajectoryDataWithMemory(
        goal=goal,
        calculator_url=calculator_url,
        site_urls=site_urls,
        steps=steps,
        reward=traj_data["reward"],
        terminated=True,
        truncated=False,
    )
    with open(traj_save_dir / f"{traj_key}.json", "w", encoding="utf-8") as traj_file:
        json.dump(trajectory.to_json(), traj_file, indent=2)


def save_if_shortest_trajectory(
    traj_key,
    traj_data,
    domain,
    goal,
    steps,
    traj_save_dir,
    best_metadata_by_intent,
    num_duplicate_traj,
):
    if should_skip_trajectory(traj_data, domain, steps):
        return False

    num_steps = len(steps)
    existing_metadata = best_metadata_by_intent.setdefault(goal, [])
    if num_duplicate_traj == 0:
        return False
    if (
        num_duplicate_traj != -1
        and len(existing_metadata) >= num_duplicate_traj
        and existing_metadata[-1]["num_steps"] <= num_steps
    ):
        return False

    save_trajectory(traj_key, traj_data, domain, goal, steps, traj_save_dir)

    metadata = {
        "traj_id": traj_key,
        "num_steps": num_steps,
        "reward": traj_data["reward"],
        "success": traj_data["success"],
        "domain": domain,
        "intent": goal,
    }
    existing_metadata.append(metadata)
    existing_metadata.sort(key=lambda metadata: metadata["num_steps"])

    if num_duplicate_traj != -1:
        for removed_metadata in existing_metadata[num_duplicate_traj:]:
            (traj_save_dir / f"{removed_metadata['traj_id']}.json").unlink(missing_ok=True)
        del existing_metadata[num_duplicate_traj:]

    return True


def write_raw_data_to_jsonl(data_dir, output_path):
    hf_token = os.environ["HF_TOKEN"]
    dataset = load_dataset("apurvaga/go-browse-wa-raw", cache_dir=str(data_dir), token=hf_token, split="train")
    dataset = dataset.remove_columns(["png"])
    dataset.to_json(str(output_path))


def process_trajectories(raw_data_path, traj_save_dir, num_duplicate_traj):
    total_trajs = 0
    best_metadata_by_intent = {}

    with open(raw_data_path, "r") as f:
        curr_traj = 0
        traj_goal = ""
        step_num = 0
        steps = []
        prev_traj_data = None
        for line in tqdm(f, desc="Processing lines", total=total_lines):
            line_data = json.loads(line)
            traj_data = line_data["json"]["traj_data"]
            step_data = line_data["json"]["step_data"]
            domain = line_data["json"]["graph_data"]["domain"]
            traj_key = line_data["__key__"].split("-")[0]

            if curr_traj == traj_data["traj_num"] - 1:

                if save_if_shortest_trajectory(
                    prev_traj_key,
                    prev_traj_data,
                    prev_domain,
                    traj_goal,
                    steps,
                    traj_save_dir,
                    best_metadata_by_intent,
                    num_duplicate_traj,
                ):
                    total_trajs += 1
                    
                # Reset step data
                step_num = 0
                steps = []
                curr_traj += 1
                traj_goal = ""

            
            prev_traj_data = line_data["json"]["traj_data"]
            prev_step_data = line_data["json"]["step_data"]
            prev_domain = domain
            prev_traj_key = line_data["__key__"].split("-")[0]
            
            traj_goal = traj_data["goal"]
            # Skip the step if thought/action/axtree is None
            axtree = step_data["obs"]["axtree_txt"]
            reasoning = step_data["thought"]
            action = step_data["parsed_action"]
            last_action_error = step_data["obs"]["last_action_error"]

            if not (axtree is None or reasoning is None or action is None):
            
                # Create the step
                observation = Observation(
                    axtree=axtree,
                    viewport_state=None,
                    extra_element_properties=None,
                    open_pages_urls=None,
                    open_pages_titles=None,
                    active_page_index=None,
                    last_action_error=last_action_error,
                    filter_viewport=False,
                    use_tabs_info=False,
                    # Go Browse uses 80k max chars. TODO: Rename the argument to axtree_max_chars
                    axtree_max_tokens=80_000,
                )
                response = f"<think>\n{reasoning}\n</think>\n<action>\n{action}\n</action>"
                response = Response(response)
                response.reasoning = reasoning
                response.action = action
                step = StepDataWithMemory(step_num=step_num, observation=observation, response=response)
                steps.append(step)
                step_num += 1

        if prev_traj_data is not None:
            if save_if_shortest_trajectory(
                prev_traj_key,
                prev_traj_data,
                prev_domain,
                traj_goal,
                steps,
                traj_save_dir,
                best_metadata_by_intent,
                num_duplicate_traj,
            ):
                total_trajs += 1

    total_trajs = sum(len(metadata) for metadata in best_metadata_by_intent.values())
    return best_metadata_by_intent, total_trajs


def write_metadata(metadata_file_path, best_metadata_by_intent):
    with open(metadata_file_path, "w", encoding="utf-8") as metadata_f:
        for metadata_list in best_metadata_by_intent.values():
            for metadata in metadata_list:
                metadata_f.write(json.dumps(metadata) + "\n")


def main():
    args = parse_args()
    args.traj_save_dir.mkdir(parents=True, exist_ok=True)

    if args.write_to_jsonl:
        write_raw_data_to_jsonl(args.data_dir, args.raw_data_path)

    best_metadata_by_intent, total_trajs = process_trajectories(
        args.raw_data_path,
        args.traj_save_dir,
        args.num_duplicate_traj,
    )
    write_metadata(args.metadata_file_path, best_metadata_by_intent)


if __name__ == "__main__":
    main()
