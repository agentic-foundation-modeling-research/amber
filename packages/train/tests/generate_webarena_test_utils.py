import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATE_WEBARENA_PATH = REPO_ROOT / "packages" / "train" / "generate_webarena.py"
GENERATE_WEBARENA_STATE_PATH = REPO_ROOT / "packages" / "train" / "generate_webarena_state.py"


class Sample:
    def __init__(self, *, index=None, group_index=None, metadata=None, reward=None, tokens=None):
        self.index = index
        self.group_index = group_index
        self.metadata = metadata or {}
        self.reward = reward
        self.tokens = tokens or []

    def get_reward_value(self, args):
        if getattr(args, "reward_key", None):
            return self.reward[args.reward_key]
        return self.reward


class _Status:
    ABORTED = "aborted"
    COMPLETED = "completed"
    TRUNCATED = "truncated"


Sample.Status = _Status


def _module(name, **attrs):
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    return module


def _stub_modules():
    return {
        "browsergym": _module("browsergym"),
        "browsergym.core": _module("browsergym.core"),
        "browsergym.core.action": _module("browsergym.core.action"),
        "browsergym.core.action.highlevel": _module(
            "browsergym.core.action.highlevel",
            HighLevelActionSet=object,
        ),
        "context_scythe": _module("context_scythe"),
        "context_scythe.agents": _module(
            "context_scythe.agents",
            TrajectoryDataWithMemory=object,
            StepDataWithMemory=object,
            TrajectoryDataWithStateMemory=object,
            StepDataWithStateMemory=object,
            Observation=object,
            Response=object,
            Memory=object,
            SingleTurnWithMemoryPromptBuilder=object,
            SingleTurnWithStateMemoryPromptBuilder=object,
            ReasoningParseError=Exception,
            MemoryParseError=Exception,
            StateParseError=Exception,
            ActionParseError=Exception,
        ),
        "context_scythe.environment": _module(
            "context_scythe.environment",
            AsyncRemoteRolloutEnv=object,
            EnvConfig=object,
            setup_env=lambda **_: True,
            teardown_env=lambda **_: None,
        ),
        "context_scythe.environment.const": _module(
            "context_scythe.environment.const",
            SITE_URL_TEMPLATES={
                "shopping": "http://{host}:{port}",
                "shopping_admin": "http://{host}:{port}/admin",
                "reddit": "http://{host}:{port}/forums/all",
                "gitlab": "http://{host}:{port}/explore",
                "map": "http://{host}:{port}",
                "wikipedia": "http://{host}:{port}/wikipedia",
                "homepage": "http://{host}:{port}",
                "calculator": "http://{host}:{port}/calculator.html",
            },
        ),
        "miles": _module("miles"),
        "miles.rollout": _module("miles.rollout"),
        "miles.rollout.base_types": _module(
            "miles.rollout.base_types",
            RolloutFnEvalOutput=object,
            RolloutFnTrainOutput=object,
        ),
        "miles.rollout.filter_hub": _module("miles.rollout.filter_hub"),
        "miles.rollout.filter_hub.base_types": _module(
            "miles.rollout.filter_hub.base_types",
            DynamicFilterOutput=object,
            MetricGatherer=object,
            call_dynamic_filter=lambda *_, **__: None,
        ),
        "miles.rollout.sglang_rollout": _module(
            "miles.rollout.sglang_rollout",
            GenerateState=object,
            eval_rollout=None,
            generate_and_rm_group=None,
        ),
        "miles.utils": _module("miles.utils"),
        "miles.utils.async_utils": _module("miles.utils.async_utils", run=lambda coro: coro),
        "miles.utils.dumper_utils": _module("miles.utils.dumper_utils"),
        "miles.utils.http_utils": _module("miles.utils.http_utils", post=None),
        "miles.utils.misc": _module("miles.utils.misc", load_function=lambda _: None),
        "miles.utils.types": _module("miles.utils.types", Sample=Sample),
        "tito": _module("tito", Qwen3TITO=object),
    }


def load_generate_webarena(module_name: str, module_path=GENERATE_WEBARENA_PATH):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, _stub_modules()):
        spec.loader.exec_module(module)
    return module
