import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from context_scythe.datagen.sft_analyzer import Analyzer

REPO_ROOT = Path(__file__).parent.parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-step memory JSONL files for saved trajectory JSON files."
    )
    parser.add_argument(
        "trajectory_path",
        help="Trajectory JSON files or directories containing trajectory JSON files.",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        help=(
            "Directory where memory JSONL files are written. "
            "Each output file is named with the trajectory task id, for example 22095.jsonl."
        ),
    )
    parser.add_argument(
        "--few_shot_dir",
        required=True,
        help=(
            "Few shot example trajectories"
        ),
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=1,
        help="Maximum number of trajectory files to analyze concurrently.",
    )
    parser.add_argument(
        "--analysis_timeout",
        type=int,
        default=5 * 60,
        help="Timeout in seconds for analyzing one trajectory.",
    )
    parser.add_argument(
        "--task_start",
        type=int,
        default=None,
        help="Start from this trajectory index after sorting.",
    )
    parser.add_argument(
        "--task_end",
        type=int,
        default=None,
        help="Stop before this trajectory index after sorting.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Regenerate memories even when the output JSONL already exists.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.max_concurrency < 1:
        parser.error("--max_concurrency must be at least 1")

    return args


def collect_trajectory_files(raw_path: str) -> list[Path]:
    path = Path(raw_path).expanduser().resolve()
    if path.is_dir():
        return sorted(
            child
            for child in path.glob("*.json")
            if child.is_file()
        )
    if path.is_file() and path.suffix == ".json":
        return [path]
    raise FileNotFoundError(f"No trajectory JSON file or directory found at {path}")


def memory_file_for_trajectory(trajectory_file: Path, save_dir: Path) -> Path:
    return save_dir / f"{trajectory_file.stem}.jsonl"


def filter_analyzed_trajectories(
    trajectory_files: list[Path],
    save_dir: Path,
    force: bool,
) -> list[Path]:
    if force:
        return trajectory_files

    unanalyzed_files = [
        trajectory_file
        for trajectory_file in trajectory_files
        if not memory_file_for_trajectory(trajectory_file, save_dir).exists()
    ]
    skipped_count = len(trajectory_files) - len(unanalyzed_files)
    if skipped_count:
        logging.info(f"Filtered out {skipped_count} already-analyzed trajectories.")
    return unanalyzed_files


def validate_memory_step_count(trajectory_file: Path, memory_file: Path) -> bool:
    with open(trajectory_file) as f:
        trajectory = json.load(f)

    expected_steps = len(trajectory["steps"])
    with open(memory_file) as f:
        actual_steps = sum(1 for line in f if line.strip())

    if actual_steps == expected_steps:
        return True

    memory_file.unlink(missing_ok=True)
    logging.error(
        "Memory step count mismatch for %s: expected %d steps, found %d. Deleted %s",
        trajectory_file,
        expected_steps,
        actual_steps,
        memory_file,
    )
    return False


def analyze_trajectory(
    trajectory_file: Path,
    few_shot_dir: Path,
    save_dir: Path,
    analysis_timeout: int,
    force: bool,
    dry_run: bool,
) -> dict[str, str]:
    task_id = trajectory_file.stem
    save_file = memory_file_for_trajectory(trajectory_file, save_dir)

    if dry_run:
        logging.info(f"Would analyze {trajectory_file} -> {save_file}")
        return {"task_id": task_id, "status": "dry-run"}
    
    few_shot_files = list(few_shot_dir.glob("*"))
    few_shot_files = [str(f) for f in few_shot_files if str(f).endswith(".json")]

    save_file.parent.mkdir(parents=True, exist_ok=True)
    analyzer = Analyzer(trajectory_file.parent, few_shot_trajectories=few_shot_files, save_dir=save_file.parent)

    logging.info(f"Analyzing {trajectory_file} -> {save_file}")
    try:
        analyzer.analyze(task_id, REPO_ROOT, timeout=analysis_timeout)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        logging.info(f"Analyzer resulted in error. Skipping task {task_id}: {exc}")
        return {"task_id": task_id, "status": "failed"}

    if not save_file.exists():
        logging.info(f"Analyzer completed but did not create {save_file}")
        return {"task_id": task_id, "status": "missing-output"}

    if not validate_memory_step_count(trajectory_file, save_file):
        return {"task_id": task_id, "status": "invalid-output"}

    return {"task_id": task_id, "status": "analyzed"}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    trajectory_files = collect_trajectory_files(args.trajectory_path)


    save_dir = Path(args.save_dir).expanduser().resolve()
    few_shot_dir = Path(args.few_shot_dir).expanduser().resolve()
    task_start = args.task_start or 0
    task_end = args.task_end if args.task_end is not None else len(trajectory_files)
    trajectory_files = trajectory_files[task_start:task_end]
    trajectory_files = filter_analyzed_trajectories(
        trajectory_files,
        save_dir,
        force=args.force,
    )

    logging.info(f"Starting {len(trajectory_files)} trajectories ({task_start}-{task_end}). Saving to {save_dir}")

    status_counts: dict[str, int] = {}
    progress = tqdm(total=len(trajectory_files), desc="Trajectories analyzed", unit="task")
    try:
        with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
            futures = [
                executor.submit(
                    analyze_trajectory,
                    trajectory_file,
                    few_shot_dir,
                    save_dir,
                    args.analysis_timeout,
                    args.force,
                    args.dry_run,
                )
                for trajectory_file in trajectory_files
            ]
            for future in as_completed(futures):
                result = future.result()
                status = result["status"]
                status_counts[status] = status_counts.get(status, 0) + 1
                progress.update(1)
    finally:
        progress.close()

    logging.info(f"Finished memory generation: {status_counts}")


if __name__ == "__main__":
    main()
