import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_webarena_test_utils import Sample, load_generate_webarena  # noqa: E402


@pytest.fixture(scope="module")
def generate_webarena():
    return load_generate_webarena("_generate_webarena_postprocess_reward_test")


def test_normalizes_episode_rewards_and_preserves_step_order(generate_webarena):
    """Reward postprocessing normalizes complete prompt groups while preserving step order."""
    args = Namespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        n_samples_per_prompt=2,
        grpo_std_normalization=True,
        reward_key=None,
    )
    samples = [
        Sample(index=0, group_index=0, metadata={"step_num": 0, "episode_reward": 1.0}),
        Sample(index=0, group_index=0, metadata={"step_num": 1, "episode_reward": 1.0}),
        Sample(index=1, group_index=0, metadata={"step_num": 0, "episode_reward": 0.0}),
        Sample(index=1, group_index=0, metadata={"step_num": 1, "episode_reward": 0.0}),
        Sample(index=2, group_index=1, metadata={"step_num": 0, "episode_reward": 1.0}),
        Sample(index=2, group_index=1, metadata={"step_num": 1, "episode_reward": 1.0}),
        Sample(index=3, group_index=1, metadata={"step_num": 0, "episode_reward": 1.0}),
        Sample(index=3, group_index=1, metadata={"step_num": 1, "episode_reward": 1.0}),
        Sample(index=3, group_index=1, metadata={"step_num": 2, "episode_reward": 1.0}),
    ]

    raw_rewards, rewards = generate_webarena.postprocess_reward(args, samples)

    assert raw_rewards == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert rewards == pytest.approx(
        [
            0.7071057558059692,
            0.7071057558059692,
            -0.7071057558059692,
            -0.7071057558059692,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )


def test_normalizes_episode_rewards_no_std_and_preserves_step_order(generate_webarena):
    """Reward postprocessing mean-centers prompt groups when std normalization is disabled."""
    args = Namespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        n_samples_per_prompt=2,
        grpo_std_normalization=False,
        reward_key=None,
    )
    samples = [
        Sample(index=0, group_index=0, metadata={"step_num": 0, "episode_reward": 1.0}),
        Sample(index=0, group_index=0, metadata={"step_num": 1, "episode_reward": 1.0}),
        Sample(index=1, group_index=0, metadata={"step_num": 0, "episode_reward": 0.0}),
        Sample(index=1, group_index=0, metadata={"step_num": 1, "episode_reward": 0.0}),
        Sample(index=2, group_index=1, metadata={"step_num": 0, "episode_reward": 1.0}),
        Sample(index=2, group_index=1, metadata={"step_num": 1, "episode_reward": 1.0}),
        Sample(index=3, group_index=1, metadata={"step_num": 0, "episode_reward": 1.0}),
        Sample(index=3, group_index=1, metadata={"step_num": 1, "episode_reward": 1.0}),
        Sample(index=3, group_index=1, metadata={"step_num": 2, "episode_reward": 1.0}),
    ]

    raw_rewards, rewards = generate_webarena.postprocess_reward(args, samples)

    assert raw_rewards == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert rewards == pytest.approx([0.5, 0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_returns_raw_rewards_when_std_disabled(generate_webarena):
    """Reward postprocessing returns raw episode rewards when normalization is disabled."""
    args = Namespace(
        advantage_estimator="grpo",
        rewards_normalization=False,
        n_samples_per_prompt=2,
        grpo_std_normalization=True,
        reward_key=None,
    )
    samples = [
        Sample(index=0, group_index=0, metadata={"step_num": 0, "episode_reward": 1.0}),
        Sample(index=0, group_index=0, metadata={"step_num": 1, "episode_reward": 1.0}),
        Sample(index=1, group_index=0, metadata={"step_num": 0, "episode_reward": 0.0}),
        Sample(index=1, group_index=0, metadata={"step_num": 1, "episode_reward": 0.0}),
        Sample(index=2, group_index=1, metadata={"step_num": 0, "episode_reward": 1.0}),
        Sample(index=2, group_index=1, metadata={"step_num": 1, "episode_reward": 1.0}),
    ]

    raw_rewards, rewards = generate_webarena.postprocess_reward(args, samples)

    assert raw_rewards == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    assert rewards == raw_rewards


def test_uses_terminal_step_episode_reward_for_all_episode_steps(generate_webarena):
    """Reward postprocessing applies the terminal episode reward to every episode step."""
    args = Namespace(
        advantage_estimator="grpo",
        rewards_normalization=False,
        n_samples_per_prompt=1,
        grpo_std_normalization=True,
        reward_key=None,
    )
    samples = [
        Sample(index=0, group_index=0, metadata={"step_num": 0, "episode_reward": 0.0}),
        Sample(index=0, group_index=0, metadata={"step_num": 1, "episode_reward": 1.0}),
        Sample(index=0, group_index=0, metadata={"step_num": 2, "episode_reward": 0.5}),
    ]

    raw_rewards, rewards = generate_webarena.postprocess_reward(args, samples)

    assert raw_rewards == [0.5, 0.5, 0.5]
    assert rewards == raw_rewards


def test_raises_when_episode_reward_and_sample_reward_are_missing(generate_webarena):
    """Reward postprocessing raises when neither episode nor sample reward is available."""
    args = Namespace(
        advantage_estimator="grpo",
        rewards_normalization=False,
        n_samples_per_prompt=1,
        grpo_std_normalization=True,
        reward_key=None,
    )
    samples = [
        Sample(index=7, group_index=3, metadata={"step_num": 0}),
    ]

    with pytest.raises(ValueError, match="Missing WebArena reward.*index=7.*group_index=3"):
        generate_webarena.postprocess_reward(args, samples)


def test_normalization_requires_complete_prompt_group(generate_webarena):
    """Reward normalization rejects prompt groups missing expected sampled episodes."""
    args = Namespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        n_samples_per_prompt=2,
        grpo_std_normalization=True,
        reward_key=None,
    )
    samples = [
        Sample(index=0, group_index=0, metadata={"step_num": 0, "episode_reward": 1.0}),
        Sample(index=0, group_index=0, metadata={"step_num": 1, "episode_reward": 1.0}),
    ]

    with pytest.raises(ValueError, match="Expected 2 episodes for group_index=0, got 1"):
        generate_webarena.postprocess_reward(args, samples)
