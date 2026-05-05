import json
import webbrowser
from pathlib import Path
from typing import Optional

from lottery_sim.dashboard import (
    _dashboard_db_path_for_root,
    _get_dashboard_job,
    _job_snapshot,
    _job_sse_events,
    export_dashboard_snapshot,
    load_dashboard_model,
    render_dashboard_html,
    save_dashboard_config,
    start_dashboard_job,
)


FASTAPI_INSTALL_HINT = "pip install fastapi uvicorn"


def create_fastapi_app(reports_dir: Path, repo_root: Path):
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"FastAPI 服务需要先安装依赖：{FASTAPI_INSTALL_HINT}") from exc

    reports_path = Path(reports_dir)
    root_path = Path(repo_root)
    app = FastAPI(title="彩票模拟分析系统")

    @app.get("/", response_class=HTMLResponse)
    @app.get("/index.html", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(render_dashboard_html(load_dashboard_model(reports_path)))

    @app.get("/health")
    async def health():
        return {"ok": True, "server": "fastapi"}

    @app.post("/api/jobs/{action}")
    async def start_job(action: str, request: Request):
        params = request.query_params
        game_code = params.get("game", "")
        options = {
            key: value
            for key, value in params.items()
            if key != "game"
        }
        job = start_dashboard_job(action, root_path, game_code=game_code, options=options)
        status_code = 202 if job.status == "running" else 400
        return JSONResponse(_job_snapshot(job), status_code=status_code)

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str):
        job = _get_dashboard_job(job_id)
        if job is None:
            return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)
        return _job_snapshot(job)

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str):
        job = _get_dashboard_job(job_id)
        if job is None:
            return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)
        return StreamingResponse(_job_sse_events(job), media_type="text/event-stream")

    @app.post("/api/config")
    async def config(request: Request):
        body = await request.body()
        form = {}
        if body:
            from urllib.parse import parse_qs
            form = {
                key: values[0]
                for key, values in parse_qs(body.decode("utf-8")).items()
                if values
            }
        save_dashboard_config(_dashboard_db_path_for_root(root_path), form)
        return {"ok": True}

    @app.get("/api/export")
    async def export(format: str = "csv"):  # noqa: A002
        try:
            path = export_dashboard_snapshot(
                load_dashboard_model(reports_path),
                root_path / "reports" / "exports",
                format,
            )
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return {"ok": True, "path": str(path)}

    return app


def serve_fastapi_dashboard(
    reports_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
    repo_root: Optional[Path] = None,
) -> None:
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"FastAPI 服务需要先安装依赖：{FASTAPI_INSTALL_HINT}") from exc

    app = create_fastapi_app(Path(reports_dir), Path(repo_root or Path.cwd()))
    url = f"http://{host}:{port}"
    print(f"FastAPI 仪表盘地址：{url}")
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=host, port=port, log_level="info")
