import argparse
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
DEFAULT_SERVICE_URLS = {
    "wikipedia_url": "#",
}
SERVICE_URLS = DEFAULT_SERVICE_URLS.copy()

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", SERVICE_URLS)


@app.get("/scratchpad.html")
def scratchpad(request: Request):
    return templates.TemplateResponse(request, "scratchpad.html")


@app.get("/calculator.html")
def calculator(request: Request):
    return templates.TemplateResponse(request, "calculator.html")

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the WebArena homepage server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=445)
    parser.add_argument("--wikipedia-url", default=DEFAULT_SERVICE_URLS["wikipedia_url"])
    args = parser.parse_args()

    SERVICE_URLS.update(
        {
            "wikipedia_url": args.wikipedia_url,
        }
    )

    uvicorn.run(app, host=args.host, port=args.port)
