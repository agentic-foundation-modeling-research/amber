import asyncio
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_webarena_test_utils import Sample, load_generate_webarena  # noqa: E402


@pytest.fixture(scope="module")
def generate_webarena():
    return load_generate_webarena("_generate_webarena_rollout_test")


def test_assigns_vm_rollout_metadata_per_sample(generate_webarena):
    """Each sample in one prompt group gets VM-local URLs and distinct mutable ports."""
    vm = generate_webarena.WebArenaVM(
        host="10.0.0.7",
        setup_server_port=7565,
        homepage_port=7564,
        calculator_port=7564,
        map_port=443,
        wikipedia_port=444,
        mutable_port_bases=[8081, 9081],
        site_port_stride=1,
    )
    group = [
        Sample(index=0, group_index=0, metadata={"sites": ["shopping", "reddit", "map", "wikipedia"]}),
        Sample(index=1, group_index=0, metadata={"sites": ["shopping", "reddit", "map", "wikipedia"]}),
    ]

    generate_webarena.assign_webarena_rollout_metadata(group, vm)

    first = group[0].metadata["rollout_metadata"]
    second = group[1].metadata["rollout_metadata"]
    assert first["site_ports"] == {"shopping": 8081, "reddit": 9081, "map": 443, "wikipedia": 444}
    assert second["site_ports"] == {"shopping": 8082, "reddit": 9082, "map": 443, "wikipedia": 444}
    assert first["mutable_site_ports"] == {"shopping": 8081, "reddit": 9081}
    assert second["mutable_site_ports"] == {"shopping": 8082, "reddit": 9082}
    assert first["site_urls"]["map"] == "http://10.0.0.7:443"
    assert first["setup_server_url"] == "http://10.0.0.7:7565"
    assert first["homepage_url"] == "http://10.0.0.7:7564"
    assert first["calculator_url"] == "http://10.0.0.7:7564/calculator.html"


def test_assigns_mutable_ports_from_task_site_order(generate_webarena):
    """Mutable port bases are not tied to hardcoded WebArena site names."""
    vm = generate_webarena.WebArenaVM(
        host="10.0.0.7",
        setup_server_port=7565,
        homepage_port=7564,
        calculator_port=7564,
        map_port=443,
        wikipedia_port=444,
        mutable_port_bases=[8081, 9081],
        site_port_stride=1,
    )
    group = [
        Sample(index=0, group_index=0, metadata={"sites": ["shopping_admin"]}),
        Sample(index=1, group_index=0, metadata={"sites": ["gitlab", "reddit"]}),
    ]

    generate_webarena.assign_webarena_rollout_metadata(group, vm)

    first = group[0].metadata["rollout_metadata"]
    second = group[1].metadata["rollout_metadata"]
    assert first["site_ports"] == {"shopping_admin": 8081}
    assert second["site_ports"] == {"gitlab": 8082, "reddit": 9082}
    assert first["site_urls"] == {"shopping_admin": "http://10.0.0.7:8081/admin"}
    assert second["site_urls"] == {
        "gitlab": "http://10.0.0.7:8082/explore",
        "reddit": "http://10.0.0.7:9082/forums/all",
    }


def test_rejects_task_with_more_mutable_sites_than_port_bases(generate_webarena):
    """Rollout metadata assignment rejects tasks needing more mutable ports than configured."""
    vm = generate_webarena.WebArenaVM(
        host="10.0.0.7",
        setup_server_port=7565,
        homepage_port=7564,
        calculator_port=7564,
        map_port=443,
        wikipedia_port=444,
        mutable_port_bases=[8081, 9081],
        site_port_stride=1,
    )
    group = [
        Sample(index=0, group_index=0, metadata={"sites": ["shopping", "shopping_admin", "reddit"]}),
    ]

    with pytest.raises(ValueError, match="only has 2 mutable port base"):
        generate_webarena.assign_webarena_rollout_metadata(group, vm)


def test_validates_one_task_id_per_prompt_group(generate_webarena):
    """Prompt-group validation accepts samples that all share one task_id."""
    group = [
        Sample(index=0, group_index=0, metadata={"task_id": 10010}),
        Sample(index=1, group_index=0, metadata={"task_id": 10010}),
    ]

    assert generate_webarena.validate_webarena_group_task_ids(group) == 10010


def test_rejects_mixed_task_ids_in_prompt_group(generate_webarena):
    """Prompt-group validation rejects samples with mixed task_id values."""
    group = [
        Sample(index=0, group_index=0, metadata={"task_id": 10010}),
        Sample(index=1, group_index=0, metadata={"task_id": 10054}),
    ]

    with pytest.raises(ValueError, match="exactly one task_id"):
        generate_webarena.validate_webarena_group_task_ids(group)


def test_rejects_missing_task_id_in_prompt_group(generate_webarena):
    """Prompt-group validation rejects samples missing metadata task_id."""
    group = [
        Sample(index=0, group_index=0, metadata={"task_id": 10010}),
        Sample(index=1, group_index=0, metadata={}),
    ]

    with pytest.raises(ValueError, match="missing metadata\\['task_id'\\]"):
        generate_webarena.validate_webarena_group_task_ids(group)


def test_generate_uses_sample_metadata_task_id(generate_webarena, monkeypatch):
    """Generation uses the task_id from sample metadata instead of CLI defaults."""
    captured = {}

    class FakeEnvConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAsyncRemoteRolloutEnv:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def close(self):
            pass

    async def fake_agent_loop(*args, **kwargs):
        return [args[1]]

    monkeypatch.setattr(generate_webarena, "EnvConfig", FakeEnvConfig)
    monkeypatch.setattr(generate_webarena, "HighLevelActionSet", lambda _: object())
    monkeypatch.setattr(generate_webarena, "AsyncRemoteRolloutEnv", FakeAsyncRemoteRolloutEnv)
    monkeypatch.setattr(generate_webarena, "setup_env", lambda **_: True)
    monkeypatch.setattr(generate_webarena, "teardown_env", lambda **_: None)
    monkeypatch.setattr(generate_webarena, "agent_loop", fake_agent_loop)

    args = Namespace(
        task_id=10010,
        env_server_url="http://env-server",
        webarena_env_timeout_s=90,
        webarena_max_steps=1,
    )
    task_config = {
        "task_id": 10054,
        "intent": "test intent",
        "sites": ["shopping_admin"],
        "rollout_metadata": {
            "site_ports": {"shopping_admin": 8081},
            "site_urls": {"shopping_admin": "http://website:8081/admin"},
            "mutable_site_ports": {"shopping_admin": 8081},
            "mutable_site_urls": {"shopping_admin": "http://website:8081/admin"},
            "setup_server_url": "http://website:7565",
            "homepage_url": "http://website:7564",
            "calculator_url": "http://website:7564/calculator.html",
        },
    }
    sample = Sample(index=0, group_index=0, metadata=task_config)

    result = asyncio.run(generate_webarena.generate(args, sample, {}))

    assert result == [sample]
    assert captured["task_id"] == 10054
    assert captured["task_config"] is task_config


def test_generate_ignores_close_failure_and_tears_down(generate_webarena, monkeypatch):
    """Remote close timeouts should not fail an otherwise completed rollout."""
    events = []

    class FakeEnvConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAsyncRemoteRolloutEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def close(self):
            events.append("close")
            raise TimeoutError("close timed out")

    async def fake_agent_loop(*args, **kwargs):
        return [args[1]]

    def fake_setup_env(**kwargs):
        events.append("setup")
        return True

    def fake_teardown_env(**kwargs):
        events.append("teardown")

    monkeypatch.setattr(generate_webarena, "EnvConfig", FakeEnvConfig)
    monkeypatch.setattr(generate_webarena, "HighLevelActionSet", lambda _: object())
    monkeypatch.setattr(generate_webarena, "AsyncRemoteRolloutEnv", FakeAsyncRemoteRolloutEnv)
    monkeypatch.setattr(generate_webarena, "setup_env", fake_setup_env)
    monkeypatch.setattr(generate_webarena, "teardown_env", fake_teardown_env)
    monkeypatch.setattr(generate_webarena, "agent_loop", fake_agent_loop)

    args = Namespace(
        env_server_url="http://env-server",
        webarena_env_timeout_s=90,
        webarena_max_steps=1,
    )
    task_config = {
        "task_id": 10054,
        "intent": "test intent",
        "sites": ["shopping_admin"],
        "rollout_metadata": {
            "site_ports": {"shopping_admin": 8081},
            "site_urls": {"shopping_admin": "http://website:8081/admin"},
            "mutable_site_ports": {"shopping_admin": 8081},
            "mutable_site_urls": {"shopping_admin": "http://website:8081/admin"},
            "setup_server_url": "http://website:7565",
            "homepage_url": "http://website:7564",
            "calculator_url": "http://website:7564/calculator.html",
        },
    }
    sample = Sample(index=0, group_index=0, metadata=task_config)

    result = asyncio.run(generate_webarena.generate(args, sample, {}))

    assert result == [sample]
    assert events == ["setup", "close", "teardown"]


def test_generate_rollout_logs_total_tokens_after_adding_aborted_samples(generate_webarena, monkeypatch, caplog):
    """Rollout logging counts tokens after aborted samples are added back."""
    output = SimpleNamespace(
        samples=[
            [
                [Sample(index=0, group_index=0, tokens=[1, 2, 3])],
                [Sample(index=1, group_index=0, tokens=[4, 5])],
            ]
        ]
    )
    aborted_samples = [[Sample(index=2, group_index=1, tokens=[6, 7, 8, 9])]]
    added_samples = []

    class FakeDataSource:
        def get_samples(self, _count):
            return []

        def add_samples(self, samples):
            added_samples.extend(samples)

    monkeypatch.setattr(generate_webarena, "generate_rollout_async", lambda *_: (output, aborted_samples))
    monkeypatch.setattr(generate_webarena, "run", lambda result: result)

    with caplog.at_level("INFO"):
        result = generate_webarena.generate_rollout(
            Namespace(rollout_global_dataset=True),
            rollout_id=0,
            data_source=FakeDataSource(),
        )

    assert result is output
    assert added_samples == aborted_samples
    assert (
        "WebArena rollout total tokens: total_tokens=9 min_tokens=2 max_tokens=4 "
        "mean_tokens=3.00 sample_count=3 aborted_sample_count=1"
    ) in caplog.text


def test_stamp_group_rewards_copies_complete_group_map_to_each_step(generate_webarena):
    """Every stepwise sample receives the complete reward map for its prompt group."""
    args = Namespace(n_samples_per_prompt=2, reward_key=None)
    first_step = Sample(index=0, group_index=7, metadata={"episode_reward": 0.0})
    first_terminal = Sample(index=0, group_index=7, metadata={"episode_reward": 1.0})
    second_step = Sample(index=1, group_index=7, metadata={"episode_reward": 0.0})
    second_terminal = Sample(index=1, group_index=7, metadata={"episode_reward": 0.25})
    data = [[[first_step, first_terminal], [second_step, second_terminal]]]

    generate_webarena._stamp_group_rewards(args, data)

    expected = {0: 1.0, 1: 0.25}
    for sample in [first_step, first_terminal, second_step, second_terminal]:
        assert sample.metadata["group_episode_rewards"] == expected


def test_stamp_group_rewards_rejects_incomplete_group(generate_webarena):
    """Stamped reward context must contain all n_samples_per_prompt episodes."""
    args = Namespace(n_samples_per_prompt=2, reward_key=None)
    data = [[[Sample(index=0, group_index=7, metadata={"episode_reward": 1.0})]]]

    with pytest.raises(ValueError, match="Expected 2 episodes for group_index=7, got 1"):
        generate_webarena._stamp_group_rewards(args, data)


def test_stamp_group_rewards_rejects_duplicate_episode_index(generate_webarena):
    """Episode indices identify completions inside the current grouping logic."""
    args = Namespace(n_samples_per_prompt=2, reward_key=None)
    data = [
        [
            [Sample(index=0, group_index=7, metadata={"episode_reward": 1.0})],
            [Sample(index=0, group_index=7, metadata={"episode_reward": 0.0})],
        ]
    ]

    with pytest.raises(ValueError, match="Duplicate WebArena episode index=0"):
        generate_webarena._stamp_group_rewards(args, data)


def test_stamp_group_rewards_rejects_mixed_group_index(generate_webarena):
    """One prompt group cannot mix samples from different group_index values."""
    args = Namespace(n_samples_per_prompt=2, reward_key=None)
    data = [
        [
            [Sample(index=0, group_index=7, metadata={"episode_reward": 1.0})],
            [Sample(index=1, group_index=8, metadata={"episode_reward": 0.0})],
        ]
    ]

    with pytest.raises(ValueError, match="mixes group_index=7 and group_index=8"):
        generate_webarena._stamp_group_rewards(args, data)


def test_postprocess_reward_uses_stamped_group_map_after_trim(generate_webarena):
    """A trimmed surviving episode still normalizes against the complete stored group."""
    args = Namespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        n_samples_per_prompt=2,
        grpo_std_normalization=False,
        reward_key=None,
    )
    group_episode_rewards = {0: 1.0, 1: 0.0}
    group_final_rewards = {0: 1.0, 1: 0.0}
    samples = [
        Sample(
            index=0,
            group_index=7,
            metadata={
                "step_num": 0,
                "episode_reward": 1.0,
                "group_episode_rewards": group_episode_rewards,
                "group_final_rewards": group_final_rewards,
            },
        ),
        Sample(
            index=0,
            group_index=7,
            metadata={
                "step_num": 1,
                "episode_reward": 1.0,
                "group_episode_rewards": group_episode_rewards,
                "group_final_rewards": group_final_rewards,
            },
        ),
    ]

    raw_rewards, rewards = generate_webarena.postprocess_reward(args, samples)

    assert raw_rewards == [1.0, 1.0]
    assert rewards == pytest.approx([0.5, 0.5])


def test_vm_pool_serializes_prompt_groups_per_vm_with_staggered_rollouts(generate_webarena):
    """Concurrent prompt groups never share one VM, and VMs are reused after release."""
    args = Namespace(
        webarena_vms=["10.0.0.8", "10.0.0.9"],
        webarena_port_bases=[8100],
        webarena_setup_server_port=7565,
        webarena_homepage_port=7564,
        webarena_map_port=443,
        webarena_wikipedia_port=444,
        webarena_site_port_stride=3,
    )

    async def simulate_rollouts():
        pool = generate_webarena.build_webarena_vm_pool(args)
        active_hosts = set()
        max_concurrent = 0
        lease_events = []
        lock = asyncio.Lock()
        durations = [0.03, 0.01, 0.02, 0.005, 0.015]

        async def run_prompt_group(group_id: int, duration: float):
            async with pool.lease() as vm:
                async with lock:
                    assert vm.host not in active_hosts
                    active_hosts.add(vm.host)
                    lease_events.append(("start", group_id, vm.host))
                    nonlocal max_concurrent
                    max_concurrent = max(max_concurrent, len(active_hosts))

                await asyncio.sleep(duration)

                async with lock:
                    assert vm.host in active_hosts
                    active_hosts.remove(vm.host)
                    lease_events.append(("finish", group_id, vm.host))

        await asyncio.gather(
            *(run_prompt_group(group_id, duration) for group_id, duration in enumerate(durations))
        )

        assert active_hosts == set()
        assert pool.available_count == 2
        assert max_concurrent == 2
        assert len([event for event in lease_events if event[0] == "start"]) == len(durations)
        assert {host for _, _, host in lease_events} == {"10.0.0.8", "10.0.0.9"}

        vm_a = await pool.acquire()
        vm_b = await pool.acquire()
        assert vm_a.mutable_port_for_site("shopping", 2, 0) == 8106
        assert vm_b.mutable_port_for_site("shopping", 2, 0) == 8106

    asyncio.run(simulate_rollouts())
