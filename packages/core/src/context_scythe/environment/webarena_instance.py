import logging
import os
import time
from urllib.parse import urljoin

import playwright.sync_api
import requests

from .const import ACCOUNTS

logger = logging.getLogger(__name__)


class CustomWebArenaInstance:
    """
    Utility class to access a WebArena instance.

    """

    def __init__(
        self,
        homepage_url: str,
        site_urls: dict,
    ) -> None:

        self.urls = site_urls
        self.home_url = homepage_url

        self.credentials = ACCOUNTS
        
        if self.home_url is not None:
            self.calculator_url = urljoin(f"{self.home_url.rstrip('/')}/", "calculator.html")

    def warmup(self):
        """
        Check the status of the instance. Raises an error if the instance is not ready to be used.
        """
        # warm-start the instance (navigate to every domain)
        retries_left = 3
        while retries_left:
            retries_left -= 1
            try:
                self._check_is_reachable(
                    timeout=60
                )  # 60 seconds, warming up after reset might be slow
                break
            except Exception as e:
                if not retries_left:
                    raise
                logger.info(
                    f"Instance unresponsive after reset, retrying ({retries_left} retries left)\n{e}"
                )

    def _check_is_reachable(self, timeout: int):
        """
        Test that every website is reachable.

        """
        for site, url in self.urls.items():
            if url == "todo":
                continue
            try:
                requests.get(url, timeout=timeout)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                raise RuntimeError(
                    f'WebArena site "{site}" ({url}) is not reacheable. Please check the URL.'
                )

    def ui_login(self, site: str, page: playwright.sync_api.Page):
        """
        Should only be called once per site (expects user to be logged out).
        """

        url = self.urls[site]

        # open a new page (tab) to perform the login
        page = page.context.new_page()
        # set long timeout (45s) as coldstart containers are slow
        page.set_default_timeout(45000)
        page.set_default_navigation_timeout(45000)

        match site:
            case "reddit":
                username = self.credentials[site]["username"]
                password = self.credentials[site]["password"]

                page.goto(f"{url}", wait_until="domcontentloaded", timeout=45000)
                page.get_by_role("link", name="Log in").click(timeout=45000)
                page.get_by_label("Username").fill(username)
                page.get_by_label("Password").fill(password)
                page.get_by_role("button", name="Log in").click()

            case "gitlab":
                username = self.credentials[site]["username"]
                password = self.credentials[site]["password"]

                page.goto(f"{url}/users/sign_in", wait_until="domcontentloaded", timeout=45000)
                page.get_by_label("Username or email").fill(username, timeout=45000)
                page.get_by_label("Password").fill(password)
                page.get_by_role("button", name="Sign in").click()

            case "shopping":
                username = self.credentials[site]["username"]
                password = self.credentials[site]["password"]

                page.goto(f"{url}/customer/account/login/", wait_until="domcontentloaded", timeout=45000)
                page.get_by_label("Email", exact=True).fill(username, timeout=45000)
                page.get_by_label("Password", exact=True).fill(password)
                page.get_by_role("button", name="Sign In").click()

            case "shopping_admin":
                username = self.credentials[site]["username"]
                password = self.credentials[site]["password"]

                page.goto(url, wait_until="domcontentloaded", timeout=45000)
                page.get_by_label("Username").fill(username, timeout=45000)
                page.get_by_label("Password").fill(password)
                page.get_by_role("button", name="Sign in").click()

            case "wikipedia":
                page.goto(url, wait_until="load", timeout=45000)

            case "map":
                page.goto(url, wait_until="load", timeout=45000)

            case _:
                raise ValueError

        # release login page
        page.close()
