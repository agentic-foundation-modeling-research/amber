from dataclasses import dataclass
import requests
import time

import logging
from .const import FIXED_SITES, MUTABLE_SITES


@dataclass(frozen=True)
class EnvConfig:
    name: str # shopping, shopping_admin, reddit, gitlab, map, wikipedia
    port: int # port where the wnvironment website is hosted
    endpoint_url: str # url of the website


def request_json(method: str, url: str, payload: dict | None = None) -> dict:
    response = requests.request(method, url, json=payload, timeout=30)
    raise_for_setup_server_error(response)
    return response.json() if response.content else {}


def raise_for_setup_server_error(response: requests.Response) -> None:
    if response.status_code < 400:
        return

    server_message = setup_server_message(response)
    error_message = (
        f"{response.status_code} Error: {response.reason} for url: {response.url}"
    )
    if server_message:
        error_message = f"{error_message}. Setup server response: {server_message}"

    raise requests.HTTPError(error_message, response=response)


def setup_server_message(response: requests.Response) -> str:
    try:
        response_json = response.json()
    except ValueError:
        return response.text.strip()

    if isinstance(response_json, dict) and response_json.get("message"):
        return str(response_json["message"])

    return response.text.strip()


def payload_for(env_config: EnvConfig) -> dict:
    return {"port": env_config.port, "service": env_config.name}


def get_port_status(setup_server_url: str, port: int) -> dict:
    status_url = f"{setup_server_url}/status?port={port}"
    status = request_json("GET", status_url)
    return status


def wait_for_port_status(
    setup_server_url: str,
    target_status_by_port: dict[int, str],
    timeout_seconds: int,
    poll_interval_seconds: int = 5,
):
    deadline = time.monotonic() + timeout_seconds
    pending_status_by_port = dict(target_status_by_port)

    while pending_status_by_port and time.monotonic() < deadline:
        check_pending_port_statuses(setup_server_url, pending_status_by_port)
        if pending_status_by_port:
            time.sleep(poll_interval_seconds)

    return not pending_status_by_port


def check_pending_port_statuses(
    setup_server_url: str,
    pending_status_by_port: dict[int, str],
) -> None:
    for port, target_status in list(pending_status_by_port.items()):
        status = get_port_status(
            setup_server_url=setup_server_url,
            port=port
        )
        if status.get("status") == target_status:
            del pending_status_by_port[port]


def port_targets(env_configs: list[EnvConfig], target_status: str) -> dict[int, str]:
    return {
        env_config.port: target_status
        for env_config in env_configs
    }


def setup_env(
    setup_server_url: str,
    env_configs: list[EnvConfig],
    setup_timeout_seconds: int = 180, # Should never take > 3 minutes
    poll_interval_seconds: int = 5,
) -> bool:
    for env_config in env_configs:

        if env_config.name in FIXED_SITES:
            logging.info(f"{env_config} on {setup_server_url}: Fixed site, does not need setup.")
            continue

        # Make the setup request for all envs
        logging.info(f"{env_config} on {setup_server_url}: Initiating setup.")
        request_json(
            "POST",
            f"{setup_server_url}/setup",
            payload_for(env_config),
        )

    # Wait for the ports to be ready
    configs_for_port_check = [env_config for env_config in env_configs if env_config.name in MUTABLE_SITES]
    if len(configs_for_port_check) == 0:
        logging.info(f"{setup_server_url}: All sites are fixed, not port check required.")
        return True
    
    target_status_by_port = port_targets(configs_for_port_check, "occupied")
    logging.info(f"{target_status_by_port} on {setup_server_url}: Waiting for setup to complete.")
    ready = wait_for_port_status(
        setup_server_url, 
        target_status_by_port,
        timeout_seconds=setup_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    return ready


def teardown_env(
    setup_server_url: str,
    env_configs: list[EnvConfig],
    teardown_timeout_seconds: int = 60, # Should never take > 1 minute
    poll_interval_seconds: int = 5,
) -> bool:
    for env_config in env_configs:
        if env_config.name in FIXED_SITES:
            logging.info(f"{env_config} on {setup_server_url}: Fixed site, does not need teardown.")
            continue

        # Make the teardown request for all envs
        logging.info(f"{env_config} on {setup_server_url}: Initiating teardown.")
        request_json(
            "POST",
            f"{setup_server_url}/teardown",
            payload_for(env_config),
        )

    # Wait for the ports to be ready
    configs_for_port_check = [env_config for env_config in env_configs if env_config.name in MUTABLE_SITES]
    if len(configs_for_port_check) == 0:
        logging.info(f"{setup_server_url}: All sites are fixed, not port check required.")
        return True
    target_status_by_port = port_targets(configs_for_port_check, "idle")
    logging.info(f"{target_status_by_port} on {setup_server_url}: Waiting for teardown to complete.")
    ready = wait_for_port_status(
        setup_server_url, 
        target_status_by_port,
        timeout_seconds=teardown_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    return ready
