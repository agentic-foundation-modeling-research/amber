import os
import logging
import json
from typing import Optional, Tuple
from pathlib import Path
import urllib.parse

import numpy as np
import playwright.sync_api

from browsergym.core.env import BrowserEnv
from browsergym.core.registration import frozen_partial
from browsergym.core.task import AbstractBrowserTask

from .webarena_instance import CustomWebArenaInstance
from .evaluation_harness import evaluator_router
from .evaluation_harness.helper_functions import reddit_normalize_url

logger = logging.getLogger(__name__)


class ConfigTaskWrapper(AbstractBrowserTask):
    """
    Create a WebArena-like task from the config instead of implicit config loading as done in GenericWebArenaTask
    """
    def __init__(
        self,
        seed: int,
        task_config: Optional[dict] = None, # We don't want this to be None, but make Optional to be compatible with BGym
        task_id: Optional[int] = None, # We don't want this to be None
        homepage_url=None, # We don't want this to be None
        site_urls: Optional[dict] = None, # We don't want this to be None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        
        assert task_id is not None
        assert homepage_url is not None
        assert site_urls is not None

        # Setup from AbstractBrowserTask
        # initiate a random number generator
        self.random = np.random.RandomState(seed)

        # task properties, will be used to set up the browsergym environment
        self.locale = None  # see https://playwright.dev/python/docs/api/class-browser#browser-new-context-option-locale
        self.timezone_id = None  # see https://playwright.dev/python/docs/api/class-browser#browser-new-context-option-timezone-id

        # task properties, will be used to set up the browsergym environment
        self.viewport = {"width": 1280, "height": 980}
        self.slow_mo = 1000  # ms
        self.timeout = 45000  # ms

        # Ensure that the URLs for all the required sites in present
        assert all([site in site_urls.keys() for site in task_config["sites"]])

        self.webarena_instance = CustomWebArenaInstance(
            homepage_url=homepage_url, site_urls=site_urls
        )

        all_configs_str = json.dumps(task_config)

        # substitute URLs
        for url_key, pattern in {
            "gitlab": "__GITLAB__",
            "reddit": "__REDDIT__",
            "shopping": "__SHOPPING__",
            "shopping_admin": "__SHOPPING_ADMIN__",
            "wikipedia": "__WIKIPEDIA__",
            "map": "__MAP__",
        }.items():
            if url_key in self.webarena_instance.urls:
                all_configs_str = all_configs_str.replace(pattern, self.webarena_instance.urls[url_key])

        # load all task configs to JSON
        config = json.loads(all_configs_str)

        self.base_url = base_url
        self.api_key = api_key
        
        self.config = config

    def setup(self, page: playwright.sync_api.Page) -> tuple[str, dict]:

        # Warm up each site
        self.webarena_instance.warmup()

        # build the evaluator
        self.evaluator = evaluator_router(self.config, base_url=self.base_url, api_key=self.api_key)

        # add extra context headers if they are present (e.g. for access token to the self hosted webarena verified instances)
        if os.environ.get("PW_EXTRA_HEADERS"):
            extra_headers_file_path = Path(os.environ["PW_EXTRA_HEADERS"])
            try:
                with open(extra_headers_file_path, "r") as f:
                    extra_headers = json.load(f)
                page.context.set_extra_http_headers(extra_headers)
            except Exception as e:
                logger.warning(
                    f"Failed to load extra headers from {extra_headers_file_path}: {e}. Make sure to set the PW_EXTRA_HEADERS environment variable to the path of an existing json file containing the extra headers. Continuing without extra headers."
                )

        # authenticate
        for site in self.config["sites"]:
            for retry_num in range(3):
                try:
                    self.webarena_instance.ui_login(site=site, page=page)
                    break
                except Exception as e:
                    if retry_num == 2:
                        raise e
                    logger.info(f"Attempt {retry_num+1}/{3}: Login failed, retrying.")
        

        # set geolocation
        geolocation = self.config.get("geolocation", None)
        page.context.set_geolocation(geolocation)

        # navigate to the starting url(s) (might need several pages)
        # https://github.com/web-arena-x/webarena/blob/c6475f0e9affe5252a2966e26b8cb4c834a4ae40/browser_env/envs.py#L150
        if self.config["start_url"]:
            start_urls = self.config["start_url"].split(" |AND| ")
            for i, url in enumerate(start_urls):
                page.goto(url, timeout=45000)
                if i < len(start_urls) - 1:
                    page = page.context.new_page()

        # recover goal
        goal = self.config["intent"]

        return goal, {}
    
    def cheat(self, page: playwright.sync_api.Page, chat_messages: list[str]) -> None:
        raise NotImplementedError

    @classmethod
    def get_task_id(cls):
        """
        Generic class for several task ids, this way of obtaining the task id is not compatible for now.
        """
        raise NotImplementedError

    def teardown(self) -> None:
        # Nothing to be done here
        # https://github.com/web-arena-x/webarena/blob/c6475f0e9affe5252a2966e26b8cb4c834a4ae40/browser_env/envs.py#L227
        pass
    
    def validate(
        self, page: playwright.sync_api.Page, chat_messages: list[str]
    ) -> Tuple[float, bool, str, dict]:

        # safeguard: check that all open tabs are either blank or within the list of WebArena URLs
        authorized_locations = ["newtab", ""] + [
            urllib.parse.urlparse(url).netloc
            for url in [*self.webarena_instance.urls.values(), self.webarena_instance.home_url]
        ]
        for open_page in page.context.pages:
            page_location = urllib.parse.urlparse(open_page.url).netloc
            if not page_location in authorized_locations:
                return 0, True, "", {"error": "Unauthorized url, terminating task"}

        # if any, use the last assistant message as the stop answer for webarena
        stop = False
        if chat_messages and chat_messages[-1]["role"] == "assistant":
            stop = True
            last_action = {"answer": chat_messages[-1]["message"]}
        elif chat_messages and chat_messages[-1]["role"] == "infeasible":
            stop = True
            last_action = {"answer": "N/A"}
        else:
            # llm_fuzzy_match() bugfix
            last_action = {"answer": "whatever"}

        # hack: fake trajectory for evaluation (only last_action["answer"] is used in the webarena evaluation codebase)
        trajectory = [{}, last_action]  # StateInfo, Action

        # call the evaluator
        try:
            score = 0.
            if stop:
                score = self.evaluator(
                    trajectory=trajectory,
                    page=page,
                    client=None,  # none of webarena's evaluators requires a cdp session
                )
        # llm_fuzzy_match() bugfix (assert "correct" in response)
        except AssertionError:
            logger.debug(
                "llm_fuzzy_match() bugfix applied: AssertionError in evaluator, using score = 0.0"
            )
            score = 0.0

        if score > 0 or stop:
            return score, True, "", {}
        else:
            return score, False, "", {}


def create_env_for_task(
        task_config: dict, 
        task_id: int, 
        homepage_url=None,
        site_urls=None,
        base_url=None,
        api_key=None,
        **env_kwargs
    ):

    task_kwargs={
        "task_config": task_config, 
        "task_id": task_id,
        "homepage_url": homepage_url,
        "site_urls": site_urls,
        "base_url": base_url,
        "api_key": api_key
    }

    # freeze task_kwargs (cannot be overriden at environment creation)
    task_entrypoint = frozen_partial(ConfigTaskWrapper, **task_kwargs)

    env = BrowserEnv(
        task_entrypoint, 
        **env_kwargs
    )

    return env
