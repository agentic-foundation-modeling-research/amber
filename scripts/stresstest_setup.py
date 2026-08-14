import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import requests


SETUP_SERVER_PORT = 7565
SERVICE1 = "shopping"
SERVICE2 = "shopping_admin"
PORTS1 = list(range(8081, 8089))
PORTS2 = list(range(9081, 9089))
POLL_INTERVAL_SECONDS = 5
SETUP_TIMEOUT_SECONDS = 600
SERVICE_CHECK_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class ServiceConfig:
    name: str
    ports: list[int]
    endpoint_path: str = ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run setup stress tests against a WebArena service host."
    )
    parser.add_argument(
        "service_host",
        help="Host or IP address where the setup server and services are reachable.",
    )
    return parser.parse_args(argv)


def setup_server_url(service_host: str) -> str:
    return f"http://{service_host}:{SETUP_SERVER_PORT}"


def request_json(method: str, url: str, payload: dict | None = None) -> dict:
    print(f"{method} {url}")
    response = requests.request(method, url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json() if response.content else {}


def payload_for(service: ServiceConfig, port: int) -> dict:
    return {"service": service.name, "port": port}


def get_port_status(setup_server_url: str, port: int) -> dict:
    status_url = f"{setup_server_url}/status?port={port}"
    status = request_json("GET", status_url)
    print(f"Port {port} status: {status}")
    return status


def wait_for_ports_status(
    setup_server_url: str,
    ports: list[int],
    target_status: str,
    timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    pending_ports = set(ports)

    while pending_ports and time.monotonic() < deadline:
        reached_ports = set()
        for port in sorted(pending_ports):
            status = get_port_status(setup_server_url, port)
            if status.get("status") == target_status:
                print(f"Port {port} reached {target_status!r}")
                reached_ports.add(port)

        pending_ports -= reached_ports
        if pending_ports:
            print(
                f"Waiting {POLL_INTERVAL_SECONDS}s for {target_status!r}; "
                f"pending ports: {sorted(pending_ports)}"
            )
            time.sleep(POLL_INTERVAL_SECONDS)

    if pending_ports:
        raise TimeoutError(
            f"Timed out waiting for ports {sorted(pending_ports)} to become {target_status!r}"
        )


def all_ports(services: list[ServiceConfig]) -> list[int]:
    return [port for service in services for port in service.ports]


def setup_services(setup_server_url: str, services: list[ServiceConfig]) -> None:
    for service in services:
        for port in service.ports:
            print(f"Starting setup for {service.name!r} on port {port}")
            setup_response = request_json(
                "POST",
                f"{setup_server_url}/setup",
                payload_for(service, port),
            )
            print(f"Setup response for port {port}: {setup_response}")


def check_service_ports(service_host: str, services: list[ServiceConfig]) -> None:
    def check_port(service: ServiceConfig, port: int) -> None:
        service_url = f"http://{service_host}:{port}{service.endpoint_path}"
        print(f"Checking service endpoint: {service_url}")
        response = requests.get(service_url, timeout=SERVICE_CHECK_TIMEOUT_SECONDS)
        response.raise_for_status()
        print(f"GET {service_url} -> {response.status_code}")

    checks = [(service, port) for service in services for port in service.ports]
    with ThreadPoolExecutor(max_workers=len(checks)) as executor:
        futures = {
            executor.submit(check_port, service, port): (service, port)
            for service, port in checks
        }
        for future in as_completed(futures):
            service, port = futures[future]
            try:
                future.result()
            except requests.RequestException as error:
                raise RuntimeError(
                    f"Service check failed for {service.name!r} on port {port}: {error}"
                ) from error


def teardown_services(setup_server_url: str, services: list[ServiceConfig]) -> None:
    for service in services:
        for port in service.ports:
            try:
                print(f"Tearing down {service.name!r} on port {port}")
                teardown_response = request_json(
                    "POST",
                    f"{setup_server_url}/teardown",
                    payload_for(service, port),
                )
                print(f"Teardown response for port {port}: {teardown_response}")
            except requests.RequestException as error:
                print(f"Teardown request failed for {service.name!r} on port {port}: {error}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_url = setup_server_url(args.service_host)
    services = [
        ServiceConfig(SERVICE1, PORTS1),
        ServiceConfig(SERVICE2, PORTS2, endpoint_path="/admin"),
    ]
    ports = all_ports(services)

    try:
        start_time = time.monotonic()
        setup_services(setup_url, services)

        print("Waiting for all setups to complete...")
        wait_for_ports_status(setup_url, ports, "occupied", SETUP_TIMEOUT_SECONDS)

        print("Checking all service endpoints...")
        check_service_ports(args.service_host, services)

        elapsed_minutes = (time.monotonic() - start_time) / 60
    except (requests.RequestException, RuntimeError, TimeoutError) as error:
        raise SystemExit(f"Setup stress test failed: {error}") from error
    finally:
        teardown_services(setup_url, services)
        try:
            print("Waiting for all teardowns to complete...")
            wait_for_ports_status(setup_url, ports, "idle", SETUP_TIMEOUT_SECONDS)
        except (requests.RequestException, TimeoutError) as error:
            print(f"Teardown wait failed: {error}")

    print(f"Setup and endpoint verification completed in {elapsed_minutes:.2f} minutes")


if __name__ == "__main__":
    main()
