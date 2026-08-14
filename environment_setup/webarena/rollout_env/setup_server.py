import enum
import logging
import subprocess
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR / ".."

ALL_PORTS = set(list(range(8081, 8089)) + list(range(9081, 9089)))
TEARDOWN_DISABLED_SERVICES = {"maps", "wikipedia"}


class ServiceName(str, enum.Enum):
    SHOPPING = "shopping"
    SHOPPING_ADMIN = "shopping_admin"
    GITLAB = "gitlab"
    REDDIT = "reddit"


class PortRequest(BaseModel):
    service: ServiceName
    port: int


class PortStatus(str, enum.Enum):
    IDLE = "idle"
    SETTING_UP = "setting_up"
    TEARING_DOWN = "tearing_down"
    ERROR = "error"
    OCCUPIED = "occupied"


class PortState:
    def __init__(self) -> None:
        self.port_locks = {port: threading.Lock() for port in ALL_PORTS}
        self.port_services = {port: None for port in ALL_PORTS}
        self.port_status = {port: PortStatus.IDLE for port in ALL_PORTS}
        self.port_errors = {port: None for port in ALL_PORTS}


port_state = PortState()


def validate_missing_scripts():
    for service in ServiceName:
        reset_script = BASE_DIR / service.value / "reset.sh"
        teardown_script = BASE_DIR / service.value / "stop_and_remove.sh"
        if not reset_script.exists():
            raise RuntimeError(f"Reset not found at {reset_script}")
        if not teardown_script.exists():
            raise RuntimeError(f"Teardown not found at {teardown_script}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_missing_scripts()
    logger.info("Using WebArena directory: %s", BASE_DIR)
    yield


app = FastAPI(lifespan=lifespan)


def _run_setup(port: int, service: str) -> None:
    try:
        reset_script = BASE_DIR / service / "reset.sh"
        container_name = f"{service}_{port}"
        result = subprocess.run(
            ["bash", str(reset_script), str(port), container_name],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Reset stdout:\n%s", result.stdout)
        port_state.port_status[port] = PortStatus.OCCUPIED
        port_state.port_errors[port] = None
    except subprocess.CalledProcessError as e:
        logger.error("Reset failed:\n%s", e.stderr)
        port_state.port_status[port] = PortStatus.ERROR
        port_state.port_errors[port] = e.stderr
    finally:
        port_state.port_locks[port].release()


def _run_teardown(port: int, service: str) -> None:
    try:
        teardown_script = BASE_DIR / service / "stop_and_remove.sh"
        container_name = f"{service}_{port}"
        result = subprocess.run(
            ["bash", str(teardown_script), container_name],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Teardown stdout:\n%s", result.stdout)
        port_state.port_status[port] = PortStatus.IDLE
        port_state.port_services[port] = None
        port_state.port_errors[port] = None
    except subprocess.CalledProcessError as e:
        logger.error("Teardown failed:\n%s", e.stderr)
        port_state.port_status[port] = PortStatus.ERROR
        port_state.port_errors[port] = e.stderr
    finally:
        port_state.port_locks[port].release()


@app.post("/setup", status_code=200)
def setup(request: PortRequest):

    port = request.port
    service = request.service

    if port not in ALL_PORTS:
        return JSONResponse({"message": f"Port {port} not allowed."}, status_code=400)

    if port_state.port_status[port] == PortStatus.OCCUPIED:
        port_service = port_state.port_services[port]
        return JSONResponse(
            content={"message": "Port Occupied", "port": port, "service": port_service},
            status_code=409,
        )

    if not port_state.port_locks[port].acquire(blocking=False):
        port_service = port_state.port_services[port]
        return JSONResponse(
            content={"message": "Setup already in progress", "port": port, "service": port_service},
            status_code=409,
        )

    port_state.port_status[port] = PortStatus.SETTING_UP
    port_state.port_services[port] = service
    port_state.port_errors[port] = None

    thread = threading.Thread(target=_run_setup, args=(port, service.value,), daemon=True)
    thread.start()

    return {"message": "Reset initiated", "port": port, "service": service}


@app.post("/teardown", status_code=200)
def teardown(request: PortRequest):

    port = request.port
    service = request.service

    if port not in ALL_PORTS:
        return JSONResponse({"message": f"Port {port} not allowed."}, status_code=400)

    if service.value in TEARDOWN_DISABLED_SERVICES:
        return JSONResponse(
            {
                "message": f"Teardown is not allowed for {service.value}.",
                "port": port,
                "service": service,
            },
            status_code=400,
        )

    port_status = port_state.port_status[port]
    port_service = port_state.port_services[port]

    if port_status not in {PortStatus.OCCUPIED, PortStatus.ERROR}:
        return JSONResponse(
            {"message": "Port not occupied", "port": port, "service": port_service},
            status_code=409,
        )

    if port_service != service:
        return JSONResponse(
            {
                "message": "Port occupied by different service",
                "port": port,
                "service": port_service,
            },
            status_code=409,
        )

    if not port_state.port_locks[port].acquire(blocking=False):
        return JSONResponse(
            content={
                "message": "Teardown already in progress",
                "port": port,
                "service": port_service,
            },
            status_code=409,
        )

    port_state.port_status[port] = PortStatus.TEARING_DOWN
    port_state.port_errors[port] = None

    thread = threading.Thread(target=_run_teardown, args=(port, service.value,), daemon=True)
    thread.start()

    return {"message": "Teardown initiated", "port": port, "service": service}


@app.get("/status")
def status(port: int):
    
    if port not in ALL_PORTS:
        return JSONResponse({"message": f"Port {port} not allowed."}, status_code=400)
    
    status = port_state.port_status[port]
    service = port_state.port_services[port]
    
    if status == PortStatus.ERROR:
        error_message = port_state.port_errors[port]
        return JSONResponse(
            {"message": "Operation failed.", "port": port, "error": error_message},
            status_code=500,
        )
    
    return JSONResponse({"status": status, "port": port, "service": service}, status_code=200)


@app.get("/health")
def health():
    return {"ok": True}
