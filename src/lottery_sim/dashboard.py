import csv
import html
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import webbrowser
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

from lottery_sim.history_db import (
    load_dashboard_config as _load_sqlite_dashboard_config,
    load_training_records as _load_sqlite_training_records,
    record_dashboard_action as _record_sqlite_dashboard_action,
    record_training_record as _record_sqlite_training_record,
    save_dashboard_config as _save_sqlite_dashboard_config,
    sync_recommendation_records,
)
from lottery_sim.issue_calendar import next_issue_from_latest_draw
from lottery_sim.recommendation_tracking import RecommendationRecord, load_recommendation_records


GAME_CONFIGS: Tuple[Tuple[str, str], ...] = (
    ("3d", "福彩3D"),
    ("pl3", "排列三"),
    ("pl5", "排列五"),
    ("qxc", "七星彩"),
    ("qlc", "七乐彩"),
    ("kl8", "快乐8"),
    ("ssq", "双色球"),
    ("dlt", "大乐透"),
)

GAME_NAMES: Dict[str, str] = dict(GAME_CONFIGS)

GAME_CSV_PATHS: Dict[str, str] = {
    "3d": "data/normalized/fucai3d.csv",
    "pl3": "data/normalized/pl3.csv",
    "pl5": "data/normalized/pl5.csv",
    "qxc": "data/normalized/qxc.csv",
    "qlc": "data/normalized/qlc.csv",
    "kl8": "data/normalized/kl8.csv",
    "ssq": "data/normalized/ssq.csv",
    "dlt": "data/normalized/dlt.csv",
}

SUMMARY_GAME_ALIASES: Dict[str, Tuple[str, ...]] = {
    "3d": ("福彩3D", "福彩3D直选"),
    "pl3": ("排列三", "排列三直选"),
    "pl5": ("排列五", "排列五直选"),
    "qxc": ("七星彩", "7星彩"),
    "qlc": ("七乐彩",),
    "kl8": ("快乐8", "快乐8选十"),
    "ssq": ("双色球",),
    "dlt": ("大乐透",),
}

ANALYSIS_COLUMNS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "3d": (("号码频率", "number"),),
    "pl3": (("号码频率", "number"),),
    "pl5": (("号码频率", "number"),),
    "qxc": (("前六位频率", "front"), ("特别号频率", "special")),
    "qlc": (("基本号频率", "basic"), ("特别号频率", "special")),
    "kl8": (("号码频率", "numbers"),),
    "ssq": (("红球频率", "red"), ("蓝球频率", "blue")),
    "dlt": (("前区频率", "front"), ("后区频率", "back")),
}

HIT_SECTION_LABELS: Dict[str, Tuple[str, ...]] = {
    "ssq": ("红球", "蓝球"),
    "dlt": ("前区", "后区"),
    "qxc": ("前六位", "特别号"),
    "qlc": ("基本号", "特别号"),
    "kl8": ("号码",),
    "3d": ("号码",),
    "pl3": ("号码",),
    "pl5": ("号码",),
}

ACTION_HISTORY_FIELDS = (
    "created_at",
    "job_id",
    "action",
    "game_code",
    "game_name",
    "status",
    "exit_code",
    "command",
    "summary",
    "archive_dir",
)

@dataclass(frozen=True)
class DashboardCandidate:
    rank: int
    strategy: str
    number: str


@dataclass(frozen=True)
class DashboardNumberFrequency:
    number: str
    count: int
    rate: float


@dataclass(frozen=True)
class DashboardAnalysisSection:
    label: str
    frequencies: Tuple[DashboardNumberFrequency, ...]


@dataclass(frozen=True)
class DashboardRangeBucket:
    label: str
    count: int
    rate: float


@dataclass(frozen=True)
class DashboardPartitionTrend:
    label: str
    average: float


@dataclass(frozen=True)
class DashboardCandidateScore:
    rank: int
    strategy: str
    number: str
    score: int
    explanation: str


@dataclass(frozen=True)
class DashboardGameAnalysis:
    draw_count: int = 0
    latest_issue: str = ""
    latest_draw_date: str = ""
    sections: Tuple[DashboardAnalysisSection, ...] = ()
    omission_sections: Tuple[DashboardAnalysisSection, ...] = ()
    sum_ranges: Tuple[DashboardRangeBucket, ...] = ()
    partition_trends: Tuple[DashboardPartitionTrend, ...] = ()
    candidate_scores: Tuple[DashboardCandidateScore, ...] = ()


@dataclass(frozen=True)
class GameDashboard:
    code: str
    name: str
    backtest_roi: str
    backtest_draws: str
    total_bets: str
    total_payout: str
    candidates: Tuple[DashboardCandidate, ...]
    reports: Dict[str, str]
    recommendation_total: int = 0
    recommendation_checked: int = 0
    recommendation_pending: int = 0
    recommendation_winning: int = 0
    recommendation_payout: float = 0
    latest_target_issue: str = ""
    latest_prize_level: str = ""
    recent_recommendations: Tuple[RecommendationRecord, ...] = ()
    target_draw_dates: Dict[str, str] = field(default_factory=dict)
    analysis: DashboardGameAnalysis = field(default_factory=DashboardGameAnalysis)


@dataclass(frozen=True)
class DashboardActionHistoryRecord:
    created_at: str
    job_id: str
    action: str
    game_code: str
    game_name: str
    status: str
    exit_code: int
    command: str
    summary: str
    archive_dir: str = ""


@dataclass(frozen=True)
class DashboardTrainingRecord:
    created_at: str
    game_code: str
    game_name: str
    action: str
    status: str
    summary: str


@dataclass(frozen=True)
class DashboardConfig:
    llm_provider: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key_masked: str = ""


@dataclass(frozen=True)
class DashboardModel:
    reports_dir: Path
    generated_at: str
    report_count: int
    games: Tuple[GameDashboard, ...]
    recommendation_summary: str
    other_reports: Dict[str, str]
    config: DashboardConfig = field(default_factory=DashboardConfig)
    training_records: Tuple[DashboardTrainingRecord, ...] = ()
    ai_summary: str = ""


@dataclass(frozen=True)
class RecommendationBatchHistory:
    game_code: str
    game_name: str
    target_issue: str
    target_draw_date: str
    generated_at: str
    run_id: str
    count: int
    checked_count: int
    pending_count: int
    winning_count: int
    payout: float
    records: Tuple[RecommendationRecord, ...] = ()


@dataclass(frozen=True)
class DashboardActionResult:
    action: str
    ok: bool
    exit_code: int
    command: str
    output: str


@dataclass
class DashboardJob:
    job_id: str
    action: str
    game_code: str
    command: Tuple[str, ...]
    stage_labels: Tuple[str, ...]
    created_at: float = field(default_factory=time.time)
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    status: str = "running"
    exit_code: Optional[int] = None
    stage_label: str = "准备执行"
    stage_index: int = 0
    output_lines: List[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


_DASHBOARD_JOBS: Dict[str, DashboardJob] = {}
_DASHBOARD_JOBS_LOCK = threading.Lock()


def load_dashboard_model(reports_dir: Path) -> DashboardModel:
    reports_path = Path(reports_dir)
    games = tuple(_load_game_dashboard(reports_path, code, name) for code, name in GAME_CONFIGS)
    data_dir = _resolve_dashboard_data_dir(reports_path)
    db_path = data_dir / "history.sqlite3"
    config = _load_dashboard_config_model(db_path)
    training_records = _load_dashboard_training_records(db_path, data_dir)
    text_reports = sorted(reports_path.glob("*.txt")) if reports_path.exists() else []
    known_report_names = {
        f"{kind}-{code}.txt"
        for code, _ in GAME_CONFIGS
        for kind in ("backtest", "compare", "stability", "recommend", "backtest-ml", "recommend-ml")
    }
    other_reports = {
        path.name: _read_text(path)
        for path in text_reports
        if path.name not in known_report_names and path.name != "recommendation-summary.txt"
    }
    return DashboardModel(
        reports_dir=reports_path,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        report_count=len(text_reports),
        games=games,
        recommendation_summary=_read_text(reports_path / "recommendation-summary.txt"),
        other_reports=other_reports,
        config=config,
        training_records=training_records,
        ai_summary=_build_ai_summary(games, config, training_records),
    )


def save_dashboard_config(db_path: Path, config: Dict[str, str]) -> None:
    _save_sqlite_dashboard_config(Path(db_path), config)


def export_dashboard_snapshot(model: DashboardModel, output_dir: Path, export_format: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    normalized_format = export_format.lower().strip()
    if normalized_format == "csv":
        path = output_path / f"dashboard-summary-{stamp}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "game_code",
                "game_name",
                "latest_target_issue",
                "recommendation_total",
                "checked",
                "pending",
                "winning",
                "payout",
            ])
            for game in model.games:
                writer.writerow([
                    game.code,
                    game.name,
                    game.latest_target_issue,
                    game.recommendation_total,
                    game.recommendation_checked,
                    game.recommendation_pending,
                    game.recommendation_winning,
                    game.recommendation_payout,
                ])
        return path
    if normalized_format == "html":
        path = output_path / f"dashboard-summary-{stamp}.html"
        path.write_text(render_dashboard_html(model), encoding="utf-8")
        return path
    raise ValueError("export_format must be csv or html")


def run_dashboard_action(
    action: str,
    repo_root: Path,
    game_code: str = "",
    options: Optional[Dict[str, str]] = None,
    runner: Optional[Callable[[Sequence[str], Path], Tuple[int, str, str]]] = None,
) -> DashboardActionResult:
    valid_game_codes = {code for code, _ in GAME_CONFIGS}
    if not game_code:
        return DashboardActionResult(
            action=action,
            ok=False,
            exit_code=2,
            command="",
            output="Missing game code. Use one game tab at a time.",
        )
    if game_code not in valid_game_codes:
        return DashboardActionResult(
            action=action,
            ok=False,
            exit_code=2,
            command="",
            output=f"Unsupported game code: {game_code}",
        )

    command = _dashboard_command(action, game_code, options or {})
    if command is None:
        return DashboardActionResult(
            action=action,
            ok=False,
            exit_code=2,
            command="",
            output=f"Unknown action: {action}",
        )

    active_runner = runner or _default_command_runner
    exit_code, stdout, stderr = active_runner(command, Path(repo_root))
    output = "\n".join(part for part in (stdout.strip(), stderr.strip()) if part)
    return DashboardActionResult(
        action=action,
        ok=exit_code == 0,
        exit_code=exit_code,
        command=" ".join(command),
        output=output,
    )


def _dashboard_command(action: str, game_code: str, options: Dict[str, str]) -> Optional[List[str]]:
    if action == "update-data":
        return [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            "scripts/update_data.ps1",
            "-Game",
            game_code,
        ]

    if action not in {"daily", "generate"}:
        return None

    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "scripts/daily.ps1",
        "-Game",
        game_code,
    ]
    if action == "generate":
        command.append("-SkipNormalize")
    command.extend(_dashboard_daily_options(game_code, options))
    return command


def _dashboard_daily_options(game_code: str, options: Dict[str, str]) -> List[str]:
    option_specs = [
        ("count", "-Count", ""),
        ("mlMinHistory", "-MlMinHistory", "30"),
        ("mlMinTrain", "-MlMinTrain", ""),
        ("mlTrainEpochs", "-MlTrainEpochs", ""),
        ("mlBacktestLimit", "-MlBacktestLimit", ""),
        ("mlBacktestEpochs", "-MlBacktestEpochs", ""),
        ("mlRetrainEvery", "-MlRetrainEvery", ""),
    ]

    command_options: List[str] = []
    for key, flag, default in option_specs:
        value = (options.get(key, "") or default).strip()
        if not value:
            continue
        if not value.isdigit():
            continue
        command_options.extend([flag, value])
    return command_options


def _default_command_runner(command: Sequence[str], cwd: Path) -> Tuple[int, str, str]:
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_command_env(),
    )
    return completed.returncode, completed.stdout, completed.stderr


def start_dashboard_job(
    action: str,
    repo_root: Path,
    game_code: str = "",
    options: Optional[Dict[str, str]] = None,
) -> DashboardJob:
    valid_game_codes = {code for code, _ in GAME_CONFIGS}
    if not game_code:
        return _failed_dashboard_job(action, game_code, "Missing game code. Use one game tab at a time.")
    if game_code not in valid_game_codes:
        return _failed_dashboard_job(action, game_code, f"Unsupported game code: {game_code}")

    command = _dashboard_command(action, game_code, options or {})
    if command is None:
        return _failed_dashboard_job(action, game_code, f"Unknown action: {action}")

    job = _create_dashboard_job(action, game_code, command)
    _store_dashboard_job(job)
    worker = threading.Thread(target=_run_dashboard_job, args=(job, Path(repo_root)), daemon=True)
    worker.start()
    return job


def _create_dashboard_job(action: str, game_code: str, command: Sequence[str]) -> DashboardJob:
    stage_labels = _dashboard_stage_labels(action, game_code)
    return DashboardJob(
        job_id=uuid.uuid4().hex,
        action=action,
        game_code=game_code,
        command=tuple(command),
        stage_labels=stage_labels,
        stage_label=stage_labels[0] if stage_labels else "执行中",
    )


def _failed_dashboard_job(action: str, game_code: str, message: str) -> DashboardJob:
    job = _create_dashboard_job(action, game_code, ())
    with job.lock:
        job.status = "failed"
        job.exit_code = 2
        job.finished_at = time.time()
        job.stage_label = "失败"
        job.output_lines.append(message)
    return job


def _store_dashboard_job(job: DashboardJob) -> None:
    with _DASHBOARD_JOBS_LOCK:
        _DASHBOARD_JOBS[job.job_id] = job
        _prune_dashboard_jobs_locked()


def _get_dashboard_job(job_id: str) -> Optional[DashboardJob]:
    with _DASHBOARD_JOBS_LOCK:
        return _DASHBOARD_JOBS.get(job_id)


def _prune_dashboard_jobs_locked() -> None:
    if len(_DASHBOARD_JOBS) <= 50:
        return
    finished = sorted(
        (
            job
            for job in _DASHBOARD_JOBS.values()
            if job.status in {"completed", "failed"}
        ),
        key=lambda job: job.finished_at or job.created_at,
    )
    for job in finished[: len(_DASHBOARD_JOBS) - 50]:
        _DASHBOARD_JOBS.pop(job.job_id, None)


def _run_dashboard_job(job: DashboardJob, repo_root: Path) -> None:
    exit_code = _stream_command(job.command, repo_root, lambda line: _update_job_from_output_line(job, line))
    with job.lock:
        job.exit_code = exit_code
        job.finished_at = time.time()
        job.status = "completed" if exit_code == 0 else "failed"
        job.stage_label = "完成" if exit_code == 0 else "失败"
        job.stage_index = len(job.stage_labels)
    try:
        _record_dashboard_job_history(job, repo_root)
    except OSError as exc:
        with job.lock:
            job.output_lines.append(f"历史记录写入失败：{exc}")


def _record_dashboard_job_history(job: DashboardJob, repo_root: Path) -> DashboardActionHistoryRecord:
    archive_dir = _archive_dashboard_reports(job, repo_root)
    with job.lock:
        record = DashboardActionHistoryRecord(
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            job_id=job.job_id,
            action=job.action,
            game_code=job.game_code,
            game_name=GAME_NAMES.get(job.game_code, job.game_code),
            status=job.status,
            exit_code=job.exit_code if job.exit_code is not None else -1,
            command=" ".join(job.command),
            summary=_summarize_job_output(job.output_lines),
            archive_dir=archive_dir,
        )
    _append_dashboard_action_record(Path(repo_root) / "data", record)
    _record_dashboard_action_to_sqlite(repo_root, record)
    return record


def _record_dashboard_action_to_sqlite(repo_root: Path, record: DashboardActionHistoryRecord) -> None:
    payload = {
        "created_at": record.created_at,
        "job_id": record.job_id,
        "action": record.action,
        "game_code": record.game_code,
        "game_name": record.game_name,
        "status": record.status,
        "exit_code": record.exit_code,
        "command": record.command,
        "summary": record.summary,
        "archive_dir": record.archive_dir,
    }
    db_path = _dashboard_db_path_for_root(repo_root)
    _record_sqlite_dashboard_action(db_path, payload)
    if record.action in {"daily", "generate"}:
        _record_sqlite_training_record(db_path, payload)


def _archive_dashboard_reports(job: DashboardJob, repo_root: Path) -> str:
    if job.action not in {"daily", "generate"} or not job.game_code:
        return ""
    source_dir = Path(repo_root) / "reports" / "latest"
    if not source_dir.exists():
        return ""
    archive_rel = Path("data") / "report-history" / job.game_code / job.job_id
    archive_dir = Path(repo_root) / archive_rel
    archive_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    paths = list(source_dir.glob(f"*-{job.game_code}.txt"))
    summary_path = source_dir / "recommendation-summary.txt"
    if summary_path.exists():
        paths.append(summary_path)
    for path in sorted({item.resolve() for item in paths}):
        shutil.copy2(path, archive_dir / path.name)
        copied += 1
    if copied == 0:
        return ""
    return archive_rel.as_posix()


def _append_dashboard_action_record(
    data_root: Path,
    record: DashboardActionHistoryRecord,
) -> None:
    path = Path(data_root) / "dashboard-history" / "actions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ACTION_HISTORY_FIELDS)
        if should_write_header:
            writer.writeheader()
        writer.writerow({
            "created_at": record.created_at,
            "job_id": record.job_id,
            "action": record.action,
            "game_code": record.game_code,
            "game_name": record.game_name,
            "status": record.status,
            "exit_code": record.exit_code,
            "command": record.command,
            "summary": record.summary,
            "archive_dir": record.archive_dir,
        })


def _summarize_job_output(lines: Sequence[str], limit: int = 800) -> str:
    text = "\n".join(line for line in lines[-8:] if line)
    if len(text) <= limit:
        return text
    return text[-limit:]


def _stream_command(
    command: Sequence[str],
    cwd: Path,
    on_line: Callable[[str], None],
) -> int:
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_command_env(),
        bufsize=1,
    )
    if process.stdout is not None:
        for raw_line in process.stdout:
            on_line(raw_line.rstrip("\r\n"))
    return process.wait()


def _update_job_from_output_line(job: DashboardJob, line: str) -> None:
    label = _stage_label_from_output_line(line)
    with job.lock:
        if line:
            job.output_lines.append(line)
        if label and label in job.stage_labels:
            job.stage_label = label
            job.stage_index = job.stage_labels.index(label) + 1


def _job_snapshot(job: DashboardJob) -> Dict[str, Any]:
    with job.lock:
        elapsed = max(0.0, (job.finished_at or time.time()) - job.started_at)
        progress = _job_progress_percent(job)
        estimated_remaining = _job_estimated_remaining_seconds(job, elapsed, progress)
        return {
            "job_id": job.job_id,
            "action": job.action,
            "game_code": job.game_code,
            "status": job.status,
            "ok": job.status == "completed",
            "exit_code": job.exit_code,
            "command": " ".join(job.command),
            "output": "\n".join(job.output_lines),
            "stage_label": job.stage_label,
            "stage_index": job.stage_index,
            "stage_total": len(job.stage_labels),
            "progress_percent": progress,
            "elapsed_seconds": int(elapsed),
            "estimated_remaining_seconds": estimated_remaining,
        }


def _job_sse_events(
    job: DashboardJob,
    poll_interval: float = 1.0,
    max_events: Optional[int] = None,
):
    emitted = 0
    while True:
        snapshot = _job_snapshot(job)
        yield f"data: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
        emitted += 1
        if snapshot["status"] in {"completed", "failed"}:
            return
        if max_events is not None and emitted >= max_events:
            return
        time.sleep(poll_interval)


def _job_progress_percent(job: DashboardJob) -> int:
    if job.status in {"completed", "failed"}:
        return 100
    total = max(1, len(job.stage_labels))
    current = max(1, job.stage_index or 1)
    return min(95, max(1, int((current - 0.2) / total * 100)))


def _job_estimated_remaining_seconds(job: DashboardJob, elapsed: float, progress: int) -> Optional[int]:
    if job.status in {"completed", "failed"}:
        return 0
    if progress <= 1 or elapsed < 1:
        return None
    estimated_total = elapsed / (progress / 100)
    return max(0, int(estimated_total - elapsed))


def _dashboard_stage_labels(action: str, game_code: str) -> Tuple[str, ...]:
    if action == "update-data":
        return ("更新开奖数据", "校验历史推荐", "更新长期汇总")
    if action not in {"daily", "generate"}:
        return ("执行中",)

    labels: List[str] = []
    game_name = GAME_NAMES.get(game_code, game_code)
    if action == "daily":
        labels.append("更新开奖数据")
    labels.extend([
        "随机基线回测",
        "多策略对比",
        "稳定性分析",
        "生成统计候选",
    ])
    labels.extend([
        f"训练{game_name}机器学习模型",
        f"{game_name}机器学习回测",
        f"生成{game_name}机器学习候选",
        f"保存{game_name}机器学习推荐",
    ])
    labels.extend([
        "校验历史推荐",
        "保存本期推荐",
        "更新长期汇总",
    ])
    return tuple(labels)


def _stage_label_from_output_line(line: str) -> str:
    match = re.match(r"\[([^\]]+)\]", line.strip())
    if not match:
        return ""
    name = match.group(1)
    if name.startswith("update-"):
        return "更新开奖数据"
    ml_match = re.match(r"(train|backtest|recommend|record)(?:-recommend)?-ml-([a-z0-9]+)", name)
    if ml_match:
        action_name, game_code = ml_match.groups()
        game_name = GAME_NAMES.get(game_code, game_code)
        if action_name == "train":
            return f"训练{game_name}机器学习模型"
        if action_name == "backtest":
            return f"{game_name}机器学习回测"
        if action_name == "recommend":
            return f"生成{game_name}机器学习候选"
        if action_name == "record":
            return f"保存{game_name}机器学习推荐"
    if name.startswith("backtest-"):
        return "随机基线回测"
    if name.startswith("compare-"):
        return "多策略对比"
    if name.startswith("stability-"):
        return "稳定性分析"
    if name.startswith("recommend-"):
        return "生成统计候选"
    if name.startswith("verify-"):
        return "校验历史推荐"
    if name.startswith("record-"):
        return "保存本期推荐"
    if name.startswith("summarize-recommendations"):
        return "更新长期汇总"
    return ""


def _command_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.setdefault("PYTHONPATH", "src")
    return env


def render_dashboard_html(model: DashboardModel) -> str:
    return "\n".join([
        "<!doctype html>",
        "<html lang=\"zh-CN\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<title>彩票模拟分析仪表盘</title>",
        "<style>",
        _dashboard_css(),
        "</style>",
        "</head>",
        "<body>",
        "<main class=\"shell\">",
        _render_header(model),
        _render_actions(model.games),
        _render_global_recommendation_history(model.games),
        _render_backend_panel(model),
        _render_lottery_tabs(model.games),
        _render_lottery_panes(model),
        "</main>",
        "<script>",
        _dashboard_js(),
        "</script>",
        "</body>",
        "</html>",
    ])


def serve_dashboard(
    reports_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
    repo_root: Optional[Path] = None,
) -> None:
    handler = _make_dashboard_handler(Path(reports_dir), Path(repo_root or Path.cwd()))
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{server.server_port}"
    print(f"仪表盘地址：{url}")
    print("按 Ctrl+C 停止服务")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("仪表盘服务已停止")
    finally:
        server.server_close()


def _load_game_dashboard(reports_path: Path, code: str, name: str) -> GameDashboard:
    backtest = _read_text(reports_path / f"backtest-{code}.txt")
    recommend = _read_text(reports_path / f"recommend-{code}.txt")
    ml_backtest = _read_text(reports_path / f"backtest-ml-{code}.txt")
    ml_recommend = _read_text(reports_path / f"recommend-ml-{code}.txt")
    metric_report = ml_backtest or backtest
    candidates = _merge_candidates(
        _parse_candidates(ml_recommend),
        _parse_candidates(recommend),
    )
    reports = {
        "回测": backtest,
        "策略对比": _read_text(reports_path / f"compare-{code}.txt"),
        "稳定性": _read_text(reports_path / f"stability-{code}.txt"),
        "候选": recommend,
        "机器学习回测": ml_backtest,
        "机器学习候选": ml_recommend,
    }
    records = _load_dashboard_recommendation_records(reports_path, code)
    record_stats = _recommendation_record_stats(records)
    draw_dates = _load_dashboard_draw_dates(reports_path, code)
    target_draw_dates = _recommendation_target_draw_dates(records, draw_dates, code)
    analysis = _load_dashboard_game_analysis(reports_path, code, candidates)
    return GameDashboard(
        code=code,
        name=name,
        backtest_roi=_parse_metric(metric_report, "返奖率"),
        backtest_draws=_parse_metric(metric_report, "回测期数"),
        total_bets=_parse_metric(metric_report, "投注注数"),
        total_payout=_strip_money(_parse_metric(metric_report, "中奖金额")),
        candidates=tuple(candidates),
        reports=reports,
        recommendation_total=record_stats["total"],
        recommendation_checked=record_stats["checked"],
        recommendation_pending=record_stats["pending"],
        recommendation_winning=record_stats["winning"],
        recommendation_payout=record_stats["payout"],
        latest_target_issue=record_stats["latest_target_issue"],
        latest_prize_level=record_stats["latest_prize_level"],
        recent_recommendations=_recent_recommendation_records(records),
        target_draw_dates=target_draw_dates,
        analysis=analysis,
    )


def _load_dashboard_recommendation_records(reports_path: Path, game_code: str) -> Tuple[RecommendationRecord, ...]:
    root = _resolve_recommendation_dir(reports_path)
    game_dir = root / game_code
    if not game_dir.exists():
        return ()
    records: List[RecommendationRecord] = []
    for path in sorted(game_dir.glob("*.csv")):
        records.extend(load_recommendation_records(path))
    if records:
        sync_recommendation_records(_dashboard_db_path_for_reports(reports_path), records)
    return tuple(records)


def _resolve_recommendation_dir(reports_path: Path) -> Path:
    reports_path = Path(reports_path)
    if reports_path.name == "latest" and reports_path.parent.name == "reports":
        return reports_path.parent.parent / "data" / "recommendations"
    return Path("data/recommendations")


def _resolve_dashboard_data_dir(reports_path: Path) -> Path:
    reports_path = Path(reports_path)
    if reports_path.name == "latest" and reports_path.parent.name == "reports":
        return reports_path.parent.parent / "data"
    return Path("data")


def _dashboard_db_path_for_reports(reports_path: Path) -> Path:
    return _resolve_dashboard_data_dir(reports_path) / "history.sqlite3"


def _dashboard_db_path_for_root(repo_root: Path) -> Path:
    return Path(repo_root) / "data" / "history.sqlite3"


def _load_dashboard_config_model(db_path: Path) -> DashboardConfig:
    config = _load_sqlite_dashboard_config(db_path)
    return DashboardConfig(
        llm_provider=config.get("llm_provider", ""),
        llm_base_url=config.get("llm_base_url", ""),
        llm_model=config.get("llm_model", ""),
        llm_api_key_masked=config.get("llm_api_key_masked", ""),
    )


def _load_dashboard_training_records(db_path: Path, data_dir: Optional[Path] = None) -> Tuple[DashboardTrainingRecord, ...]:
    records = _load_sqlite_training_records(db_path)
    loaded = [
        DashboardTrainingRecord(
            created_at=str(record.get("created_at", "")),
            game_code=str(record.get("game_code", "")),
            game_name=str(record.get("game_name", "")),
            action=str(record.get("action", "")),
            status=str(record.get("status", "")),
            summary=str(record.get("summary", "")),
        )
        for record in records
    ]
    if not loaded and data_dir is not None:
        for record in _load_dashboard_action_history(data_dir):
            if record.action not in {"daily", "generate"}:
                continue
            loaded.append(DashboardTrainingRecord(
                created_at=record.created_at,
                game_code=record.game_code,
                game_name=record.game_name,
                action=record.action,
                status=record.status,
                summary=record.summary,
            ))
    return tuple(loaded[:50])


def _build_ai_summary(
    games: Sequence[GameDashboard],
    config: DashboardConfig,
    training_records: Sequence[DashboardTrainingRecord],
) -> str:
    configured = "已配置" if config.llm_base_url and config.llm_model else "未配置"
    total_recommendations = sum(game.recommendation_total for game in games)
    pending = sum(game.recommendation_pending for game in games)
    winning = sum(game.recommendation_winning for game in games)
    latest_training = training_records[0].created_at if training_records else "暂无"
    return (
        f"大模型接口{configured}；当前推荐记录{total_recommendations}条，"
        f"待开奖{pending}条，中奖记录{winning}条；最近训练/生成记录：{latest_training}。"
        "未配置 API 时使用本地规则总结，配置后可扩展为真实大模型报告。"
    )


def _load_dashboard_draw_dates(reports_path: Path, game_code: str) -> Dict[str, str]:
    path = _resolve_history_data_dir(reports_path) / Path(GAME_CSV_PATHS[game_code]).name
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {
            row.get("issue", ""): row.get("draw_date", "")
            for row in csv.DictReader(f)
            if row.get("issue") and row.get("draw_date")
        }


def _load_dashboard_game_analysis(
    reports_path: Path,
    game_code: str,
    candidates: Sequence[DashboardCandidate] = (),
) -> DashboardGameAnalysis:
    path = _resolve_history_data_dir(reports_path) / Path(GAME_CSV_PATHS[game_code]).name
    if not path.exists():
        return DashboardGameAnalysis()

    counters = {
        label: Counter()
        for label, _ in ANALYSIS_COLUMNS.get(game_code, ())
    }
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("issue"):
                continue
            rows.append(row)
            for label, column in ANALYSIS_COLUMNS.get(game_code, ()):
                counters[label].update(_analysis_tokens(game_code, row.get(column, "")))

    if not rows:
        return DashboardGameAnalysis()

    latest = max(rows, key=lambda row: _issue_sort_value(row.get("issue", "")))
    sections = tuple(
        DashboardAnalysisSection(
            label=label,
            frequencies=_rank_number_frequencies(counter, len(rows)),
        )
        for label, counter in counters.items()
    )
    ordered_rows = sorted(rows, key=lambda row: _issue_sort_value(row.get("issue", "")))
    omission_sections = tuple(
        DashboardAnalysisSection(
            label=f"{label}遗漏排行",
            frequencies=_rank_number_omissions(game_code, label, column, ordered_rows, counter),
        )
        for label, column in ANALYSIS_COLUMNS.get(game_code, ())
        for counter in (counters.get(label, Counter()),)
    )
    sum_ranges = _analysis_sum_ranges(game_code, ordered_rows)
    partition_trends = _analysis_partition_trends(game_code, ordered_rows)
    candidate_scores = _analysis_candidate_scores(game_code, candidates, counters, len(rows))
    return DashboardGameAnalysis(
        draw_count=len(rows),
        latest_issue=latest.get("issue", ""),
        latest_draw_date=latest.get("draw_date", ""),
        sections=sections,
        omission_sections=omission_sections,
        sum_ranges=sum_ranges,
        partition_trends=partition_trends,
        candidate_scores=candidate_scores,
    )


def _analysis_tokens(game_code: str, value: str) -> Tuple[str, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    if game_code in {"3d", "pl3", "pl5"} and " " not in text:
        return tuple(ch for ch in text if ch.isdigit())
    return tuple(part.strip() for part in re.split(r"\s+", text) if part.strip())


def _rank_number_frequencies(counter: Counter, draw_count: int, limit: int = 12) -> Tuple[DashboardNumberFrequency, ...]:
    if not counter or draw_count <= 0:
        return ()
    ranked = sorted(counter.items(), key=lambda item: (-item[1], _issue_sort_value(item[0]), item[0]))
    return tuple(
        DashboardNumberFrequency(
            number=str(number),
            count=count,
            rate=count / draw_count,
        )
        for number, count in ranked[:limit]
    )


def _rank_number_omissions(
    game_code: str,
    label: str,
    column: str,
    rows: Sequence[Dict[str, str]],
    counter: Counter,
    limit: int = 12,
) -> Tuple[DashboardNumberFrequency, ...]:
    if not rows or not counter:
        return ()
    last_seen: Dict[str, int] = {}
    for index, row in enumerate(rows):
        for token in _analysis_tokens(game_code, row.get(column, "")):
            last_seen[token] = index
    current_index = len(rows) - 1
    ranked = sorted(
        (
            (number, current_index - last_seen.get(number, -1))
            for number in counter
        ),
        key=lambda item: (-item[1], _issue_sort_value(item[0]), item[0]),
    )
    return tuple(
        DashboardNumberFrequency(number=str(number), count=omission, rate=0)
        for number, omission in ranked[:limit]
    )


def _analysis_sum_ranges(game_code: str, rows: Sequence[Dict[str, str]]) -> Tuple[DashboardRangeBucket, ...]:
    if not rows:
        return ()
    width = 10 if game_code in {"3d", "pl3", "pl5", "qxc"} else 30
    counter: Counter = Counter()
    for row in rows:
        total = sum(_analysis_row_numbers(game_code, row))
        start = (total // width) * width
        counter[f"{start}-{start + width - 1}"] += 1
    return tuple(
        DashboardRangeBucket(label=label, count=count, rate=count / len(rows))
        for label, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:8]
    )


def _analysis_partition_trends(game_code: str, rows: Sequence[Dict[str, str]]) -> Tuple[DashboardPartitionTrend, ...]:
    if not rows:
        return ()
    max_number = _analysis_max_number(game_code)
    totals: Counter = Counter()
    for row in rows:
        for number in _analysis_row_numbers(game_code, row):
            totals[_partition_label(number, max_number)] += 1
    return tuple(
        DashboardPartitionTrend(label=label, average=totals[label] / len(rows))
        for label in ("低区", "中区", "高区")
    )


def _analysis_candidate_scores(
    game_code: str,
    candidates: Sequence[DashboardCandidate],
    counters: Dict[str, Counter],
    draw_count: int,
) -> Tuple[DashboardCandidateScore, ...]:
    if not candidates or draw_count <= 0:
        return ()
    merged_counter: Counter = Counter()
    for counter in counters.values():
        merged_counter.update(counter)
    max_number = _analysis_max_number(game_code)
    scored: List[DashboardCandidateScore] = []
    for candidate in candidates[:10]:
        numbers = _candidate_numbers(game_code, candidate.number)
        if not numbers:
            continue
        hot_ratio = sum(merged_counter.get(f"{number:02d}", merged_counter.get(str(number), 0)) for number in numbers)
        hot_ratio = hot_ratio / max(1, len(numbers) * draw_count)
        partition_counts = Counter(_partition_label(number, max_number) for number in numbers)
        balance = len([value for value in partition_counts.values() if value > 0]) / 3
        score = int(min(99, max(1, hot_ratio * 70 + balance * 30)))
        explanation = (
            f"频率均值{hot_ratio:.1%}，和值{sum(numbers)}，"
            f"分区低/中/高={partition_counts['低区']}/{partition_counts['中区']}/{partition_counts['高区']}"
        )
        scored.append(DashboardCandidateScore(
            rank=candidate.rank,
            strategy=candidate.strategy,
            number=candidate.number,
            score=score,
            explanation=explanation,
        ))
    return tuple(scored)


def _analysis_row_numbers(game_code: str, row: Dict[str, str]) -> Tuple[int, ...]:
    values: List[int] = []
    for _, column in ANALYSIS_COLUMNS.get(game_code, ()):
        for token in _analysis_tokens(game_code, row.get(column, "")):
            try:
                values.append(int(token))
            except ValueError:
                continue
    return tuple(values)


def _candidate_numbers(game_code: str, text: str) -> Tuple[int, ...]:
    value = str(text or "")
    if game_code in {"3d", "pl3", "pl5"} and " " not in value and "+" not in value:
        return tuple(int(ch) for ch in value if ch.isdigit())
    return tuple(int(match) for match in re.findall(r"\d+", value))


def _analysis_max_number(game_code: str) -> int:
    return {
        "ssq": 33,
        "dlt": 35,
        "qlc": 30,
        "kl8": 80,
        "3d": 9,
        "pl3": 9,
        "pl5": 9,
        "qxc": 9,
    }.get(game_code, 33)


def _partition_label(number: int, max_number: int) -> str:
    first = max(1, max_number // 3)
    second = max(first + 1, max_number * 2 // 3)
    if number <= first:
        return "低区"
    if number <= second:
        return "中区"
    return "高区"


def _recommendation_target_draw_dates(
    records: Sequence[RecommendationRecord],
    draw_dates: Dict[str, str],
    game_code: str,
) -> Dict[str, str]:
    target_dates: Dict[str, str] = {}
    for record in records:
        if record.target_issue in target_dates:
            continue
        if record.target_issue in draw_dates:
            target_dates[record.target_issue] = draw_dates[record.target_issue]
            continue
        history_date = draw_dates.get(record.history_until_issue, "")
        if not history_date:
            target_dates[record.target_issue] = ""
            continue
        try:
            target_dates[record.target_issue] = str(
                next_issue_from_latest_draw(game_code, record.history_until_issue, history_date).draw_date
            )
        except (ValueError, OverflowError):
            target_dates[record.target_issue] = ""
    return target_dates


def _load_dashboard_action_history(
    data_root: Path,
    limit: int = 30,
) -> Tuple[DashboardActionHistoryRecord, ...]:
    path = Path(data_root) / "dashboard-history" / "actions.csv"
    if not path.exists():
        return ()
    records: List[DashboardActionHistoryRecord] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            records.append(DashboardActionHistoryRecord(
                created_at=row.get("created_at", ""),
                job_id=row.get("job_id", ""),
                action=row.get("action", ""),
                game_code=row.get("game_code", ""),
                game_name=row.get("game_name", ""),
                status=row.get("status", ""),
                exit_code=_safe_int(row.get("exit_code", "")),
                command=row.get("command", ""),
                summary=row.get("summary", ""),
                archive_dir=row.get("archive_dir", ""),
            ))
    return tuple(sorted(records, key=lambda record: record.created_at, reverse=True)[:limit])


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _recommendation_record_stats(records: Sequence[RecommendationRecord]) -> Dict[str, Any]:
    checked = [record for record in records if record.status == "checked"]
    winning = [record for record in checked if record.payout > 0]
    latest = max(records, key=_recommendation_record_sort_key, default=None)
    latest_issue_records = [record for record in records if latest and record.target_issue == latest.target_issue]
    latest_checked = [record for record in latest_issue_records if record.status == "checked"]
    latest_pending = any(record.status == "pending" for record in latest_issue_records)
    latest_issue_winning = [record for record in latest_checked if record.payout > 0]
    latest_winning = max(latest_issue_winning, key=_recommendation_record_sort_key, default=None)
    if not latest:
        latest_prize_level = ""
    elif latest_pending or not latest_checked:
        latest_prize_level = "待开奖"
    elif latest_winning:
        latest_prize_level = latest_winning.prize_level or "已中奖"
    else:
        latest_prize_level = "未中奖"
    return {
        "total": len(records),
        "checked": len(checked),
        "pending": sum(1 for record in records if record.status == "pending"),
        "winning": len(winning),
        "payout": sum(record.payout for record in checked),
        "latest_target_issue": latest.target_issue if latest else "",
        "latest_prize_level": latest_prize_level,
    }


def _recent_recommendation_records(
    records: Sequence[RecommendationRecord],
    limit: int = 0,
) -> Tuple[RecommendationRecord, ...]:
    ordered = tuple(sorted(records, key=_recommendation_record_sort_key, reverse=True))
    if limit > 0:
        return ordered[:limit]
    return ordered


def _recommendation_record_sort_key(record: RecommendationRecord) -> Tuple[int, str, str, int]:
    return (
        _issue_sort_value(record.target_issue),
        record.generated_at,
        record.run_id,
        -record.rank,
    )


def _issue_sort_value(issue: str) -> int:
    digits = "".join(ch for ch in str(issue) if ch.isdigit())
    return int(digits) if digits else -1


def _render_header(model: DashboardModel) -> str:
    available_games = sum(1 for game in model.games if any(game.reports.values()))
    return f"""
<header class="app-header">
  <div>
    <h1>彩票模拟分析仪表盘</h1>
    <p>报告目录：{html.escape(str(model.reports_dir))}</p>
  </div>
  <dl class="header-metrics">
    <div><dt>彩种</dt><dd>{available_games}</dd></div>
    <div><dt>报告</dt><dd>{model.report_count}</dd></div>
    <div><dt>更新时间</dt><dd>{html.escape(model.generated_at)}</dd></div>
  </dl>
</header>
<section class="risk">风险提示：彩票具有强随机性，历史分析和候选号码只能用于模拟研究，不能保证命中。</section>
""".strip()


def _render_actions(games: Sequence[GameDashboard]) -> str:
    tabs = "\n".join(
        f"<button type=\"button\" class=\"game-action-tab{' active' if index == 0 else ''}\" data-game-tab=\"{html.escape(game.code)}\">{html.escape(game.name)}</button>"
        for index, game in enumerate(games)
    )
    panes = "\n".join(_render_action_pane(game, index == 0) for index, game in enumerate(games))
    return f"""
<section class="actions">
  <div class="action-head">
    <h2>本地操作</h2>
    <span>按彩种单独执行</span>
  </div>
  <nav class="game-action-tabs" aria-label="彩种操作">{tabs}</nav>
  {panes}
  <div class="action-log" id="action-log" hidden>
    <div class="action-log-head">
      <span id="action-status">就绪</span>
      <div class="action-log-actions">
        <button type="button" class="action-clear" id="refresh-dashboard-data" hidden>刷新页面数据</button>
        <button type="button" class="action-clear" id="clear-action-output">关闭</button>
      </div>
    </div>
    <p id="action-summary">等待操作</p>
    <div class="progress-panel" id="progress-panel" hidden>
      <div class="progress-head">
        <span>任务进度</span>
        <span id="progress-stage">准备执行</span>
      </div>
      <progress id="action-progress" max="100" value="0"></progress>
      <div class="progress-meta">
        <span id="progress-elapsed">已用时间：0秒</span>
        <span id="progress-remaining">预计剩余：计算中</span>
      </div>
    </div>
    <details id="action-details">
      <summary>查看执行日志</summary>
      <pre id="action-output" aria-live="polite"></pre>
    </details>
  </div>
</section>
""".strip()


def _render_global_recommendation_history(games: Sequence[GameDashboard]) -> str:
    batches = _recommendation_batches(games)
    if not batches:
        return """
<details class="panel recommendation-history-global">
  <summary class="recommendation-history-summary">
    <strong>推荐历史记录</strong>
    <span class="history-toggle history-toggle-closed">展开</span>
    <span class="history-toggle history-toggle-open">收起</span>
  </summary>
  <p class="empty">暂无已保存推荐记录。</p>
</details>
""".strip()
    rows = "\n".join(_render_recommendation_batch(batch) for batch in batches)
    return f"""
<details class="panel recommendation-history-global">
  <summary class="recommendation-history-summary">
    <strong>推荐历史记录</strong>
    <span class="history-toggle history-toggle-closed">展开</span>
    <span class="history-toggle history-toggle-open">收起</span>
  </summary>
  <p class="panel-note">按彩种、目标期和生成批次汇总；点开批次可查看当时每注推荐、开奖号码、奖级和奖金。</p>
  <div class="recommendation-batches">{rows}</div>
</details>
""".strip()


def _render_backend_panel(model: DashboardModel) -> str:
    config = model.config
    training_rows = _render_training_record_rows(model.training_records)
    return f"""
<details class="panel backend-panel">
  <summary class="recommendation-history-summary">
    <strong>后台配置</strong>
    <span>大模型、训练记录、AI总结和导出</span>
  </summary>
  <section class="backend-grid">
    <form class="backend-card" id="llm-config-form">
      <h3>大模型 API 配置</h3>
      <label>服务商<input name="llm_provider" value="{html.escape(config.llm_provider)}" placeholder="openai-compatible"></label>
      <label>Base URL<input name="llm_base_url" value="{html.escape(config.llm_base_url)}" placeholder="https://api.example.com/v1"></label>
      <label>模型名称<input name="llm_model" value="{html.escape(config.llm_model)}" placeholder="gpt-4.1-mini"></label>
      <label>API Key<input name="llm_api_key" value="" placeholder="{html.escape(config.llm_api_key_masked or '未配置')}"></label>
      <button type="submit" class="action-button">保存配置</button>
      <p class="panel-note" id="config-save-status">API Key 只保存在本地 SQLite，不会提交到 GitHub。</p>
    </form>
    <section class="backend-card">
      <h3>模型训练记录</h3>
      <table><thead><tr><th>时间</th><th>彩种</th><th>动作</th><th>状态</th></tr></thead><tbody>{training_rows}</tbody></table>
    </section>
    <section class="backend-card">
      <h3>AI 报告总结</h3>
      <p>{html.escape(model.ai_summary)}</p>
    </section>
    <section class="backend-card">
      <h3>导出 CSV/HTML</h3>
      <div class="export-actions">
        <button type="button" class="action-button" data-export="csv">导出 CSV</button>
        <button type="button" class="action-button" data-export="html">导出 HTML</button>
      </div>
      <p class="panel-note" id="export-status">导出文件会写入 reports/exports。</p>
    </section>
  </section>
</details>
""".strip()


def _render_training_record_rows(records: Sequence[DashboardTrainingRecord]) -> str:
    if not records:
        return "<tr><td colspan=\"4\">暂无训练记录</td></tr>"
    return "\n".join(
        "<tr>"
        f"<td>{html.escape(record.created_at)}</td>"
        f"<td>{html.escape(record.game_name or record.game_code)}</td>"
        f"<td>{html.escape(record.action)}</td>"
        f"<td>{html.escape(record.status)}</td>"
        "</tr>"
        for record in records[:20]
    )


def _recommendation_batches(games: Sequence[GameDashboard]) -> Tuple[RecommendationBatchHistory, ...]:
    batches: Dict[Tuple[str, str, str, str], List[RecommendationRecord]] = {}
    draw_dates: Dict[Tuple[str, str], str] = {}
    names: Dict[str, str] = {}
    for game in games:
        names[game.code] = game.name
        for issue, draw_date in game.target_draw_dates.items():
            draw_dates[(game.code, issue)] = draw_date
        for record in game.recent_recommendations:
            key = (record.game_code, record.target_issue, record.generated_at, record.run_id)
            batches.setdefault(key, []).append(record)

    result: List[RecommendationBatchHistory] = []
    for (game_code, target_issue, generated_at, run_id), records in batches.items():
        checked = [record for record in records if record.status == "checked"]
        result.append(RecommendationBatchHistory(
            game_code=game_code,
            game_name=records[0].game_name or names.get(game_code, game_code),
            target_issue=target_issue,
            target_draw_date=draw_dates.get((game_code, target_issue), ""),
            generated_at=generated_at,
            run_id=run_id,
            count=len(records),
            checked_count=len(checked),
            pending_count=sum(1 for record in records if record.status == "pending"),
            winning_count=sum(1 for record in checked if record.payout > 0),
            payout=sum(record.payout for record in checked),
            records=tuple(sorted(records, key=_recommendation_record_sort_key, reverse=True)),
        ))
    return tuple(sorted(
        result,
        key=lambda item: (_issue_sort_value(item.target_issue), item.generated_at, item.run_id),
        reverse=True,
    ))


def _render_recommendation_batch(batch: RecommendationBatchHistory) -> str:
    status = "已开奖" if batch.pending_count == 0 and batch.checked_count else "待开奖"
    detail_rows = "\n".join(_render_recommendation_batch_detail_row(record) for record in batch.records)
    return f"""
<details class="recommendation-batch">
  <summary>
    <span>{html.escape(batch.game_name)}</span>
    <span>目标期 {html.escape(batch.target_issue or '-')}</span>
    <span>开奖 {html.escape(batch.target_draw_date or '-')}</span>
    <span>生成 {html.escape(batch.generated_at or '-')}</span>
    <span>候选 {batch.count}</span>
    <span>{status}</span>
    <span>中奖 {batch.winning_count}</span>
    <span>奖金 {_format_dashboard_amount(batch.payout)}</span>
    <strong class="batch-toggle-text">展开明细</strong>
  </summary>
  <div class="batch-detail">
    <div class="batch-id">批次：{html.escape(batch.run_id or '-')}</div>
    <table>
      <thead><tr><th>排名</th><th>策略</th><th>推荐号码</th><th>开奖号码</th><th>奖级/命中</th><th>命中明细</th><th>奖金</th><th>状态</th></tr></thead>
      <tbody>{detail_rows}</tbody>
    </table>
  </div>
</details>
""".strip()


def _render_recommendation_batch_detail_row(record: RecommendationRecord) -> str:
    status = "已开奖" if record.status == "checked" else "待开奖"
    prize = record.prize_level or ("待开奖" if record.status == "pending" else "未中奖")
    hit_detail = _render_hit_detail(record)
    return (
        "<tr>"
        f"<td>{html.escape(str(record.rank))}</td>"
        f"<td>{html.escape(record.strategy_name)}</td>"
        f"<td>{html.escape(record.numbers)}</td>"
        f"<td>{html.escape(record.draw_numbers or '-')}</td>"
        f"<td>{html.escape(prize)}</td>"
        f"<td>{hit_detail}</td>"
        f"<td>{_format_dashboard_amount(record.payout)}</td>"
        f"<td>{status}</td>"
        "</tr>"
    )


def _render_action_pane(game: GameDashboard, active: bool) -> str:
    ml_controls = _render_ml_controls(game)
    return f"""
<div class="game-action-pane{' active' if active else ''}" data-game-pane="{html.escape(game.code)}">
  {ml_controls}
  <div class="action-buttons">
    <button type="button" class="action-button" data-action="update-data" data-game="{html.escape(game.code)}">更新开奖数据</button>
    <button type="button" class="action-button" data-action="generate" data-game="{html.escape(game.code)}">生成推荐报告</button>
    <button type="button" class="action-button primary" data-action="daily" data-game="{html.escape(game.code)}">更新并生成推荐</button>
  </div>
</div>
""".strip()


def _render_ml_controls(game: GameDashboard) -> str:
    ml_name = _ml_display_name(game)
    basic_controls = [
        ("候选数量", "count", "10", "生成几组候选号码。日常看 5-10 组就够，多了只会增加选择负担。"),
        ("回测期数", "mlBacktestLimit", "30", "用最近多少期历史做模拟验证。越大越慢，但更能看长期表现。"),
    ]
    advanced_controls = [
        ("特征最小历史", "mlMinHistory", "30", "构造机器学习特征前至少需要多少期历史。它不是只训练30期，而是限制最早可用训练样本。"),
        ("首次回测训练历史", "mlMinTrain", "200", "滚动回测开始前至少保留多少期用于训练。回测每一期都会使用目标期之前的可用历史。"),
        ("训练轮数", "mlTrainEpochs", "8", "模型学习历史规律的迭代次数。越大越慢，不代表越准。"),
        ("回测训练轮数", "mlBacktestEpochs", "3", "回测过程中每次重新训练模型的轮数。日常可以低一点，加快速度。"),
        ("重训间隔", "mlRetrainEvery", "10", "回测时隔多少期重新训练一次模型。数值小更贴近实时训练，但更慢。"),
    ]
    return f"""
<section class="param-panel">
  <div class="param-title">{html.escape(ml_name)}机器学习参数</div>
  <p>机器学习参数只影响模拟候选和回测速度，不代表提高中奖概率。</p>
  <p class="panel-note">回测期数改成35后，需要点“生成推荐报告”或“更新并生成推荐”重新生成机器学习回测；只点“更新开奖数据”只抓新开奖，不会重算回测。</p>
  {_render_ml_context(game)}
  <div class="basic-param-grid">{_render_param_fields(basic_controls)}</div>
  <details class="advanced-params">
    <summary>高级机器学习参数</summary>
    <div class="param-grid">{_render_param_fields(advanced_controls)}</div>
  </details>
</section>
""".strip()


def _render_param_fields(controls: Sequence[Tuple[str, str, str, str]]) -> str:
    return "\n".join(
        f"""
<label class="param-field">
  <span class="param-label">{html.escape(label)}</span>
  <input type="number" min="1" name="{html.escape(name)}" value="{html.escape(value)}" data-action-param="{html.escape(name)}">
  <span class="param-desc">{html.escape(help_text)}</span>
</label>
""".strip()
        for label, name, value, help_text in controls
    )


def _ml_display_name(game: GameDashboard) -> str:
    for candidate in game.candidates:
        if "机器学习" in candidate.strategy:
            return candidate.strategy.replace("机器学习", "").strip() or game.name
    return game.name


def _render_ml_context(game: GameDashboard) -> str:
    training_history = _parse_metric(game.reports.get("机器学习候选", ""), "历史期数")
    backtest_draws = _parse_metric(game.reports.get("机器学习回测", ""), "回测期数")
    if not training_history and not backtest_draws:
        return ""
    history_text = f"{training_history}期" if training_history else "-"
    backtest_text = f"最近{backtest_draws}期" if backtest_draws else "-"
    return f"""
<dl class="ml-context">
  <div><dt>训练历史</dt><dd>{html.escape(history_text)}</dd></div>
  <div><dt>机器学习回测</dt><dd>{html.escape(backtest_text)}</dd></div>
</dl>
""".strip()


def _format_dashboard_amount(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _compact_text(value: str, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[:limit - 1] + "…"


def _render_lottery_tabs(games: Sequence[GameDashboard]) -> str:
    buttons = "\n".join(
        f"<button type=\"button\" class=\"lottery-tab{' active' if index == _default_game_index(games) else ''}\" data-lottery-tab=\"{html.escape(game.code)}\">{html.escape(game.name)}</button>"
        for index, game in enumerate(games)
    )
    return f"<nav class=\"lottery-tabs\" aria-label=\"彩种内容\">{buttons}</nav>"


def _render_lottery_panes(model: DashboardModel) -> str:
    default_index = _default_game_index(model.games)
    panes = "\n".join(
        _render_lottery_pane(game, model.recommendation_summary, index == default_index)
        for index, game in enumerate(model.games)
    )
    return f"<section class=\"lottery-panes\">{panes}</section>"


def _render_lottery_pane(game: GameDashboard, summary_text: str, active: bool) -> str:
    return f"""
<section class="lottery-pane{' active' if active else ''}" data-lottery-pane="{html.escape(game.code)}">
  {_render_game_view_tabs(game)}
  {_render_game_overview(game)}
  {_render_game_candidates(game)}
  {_render_game_analysis(game)}
  {_render_game_summary(game, summary_text)}
  {_render_game_reports(game)}
</section>
""".strip()


def _render_game_view_tabs(game: GameDashboard) -> str:
    tabs = [
        (f"overview-{game.code}", "总览"),
        (f"candidates-{game.code}", "候选号码"),
        (f"analysis-{game.code}", "号码分析"),
        (f"summary-{game.code}", "长期汇总"),
        (f"reports-{game.code}", "原始报告"),
    ]
    buttons = "\n".join(
        f"<button type=\"button\" class=\"tab{' active' if index == 0 else ''}\" data-target=\"{view}\">{label}</button>"
        for index, (view, label) in enumerate(tabs)
    )
    return f"<nav class=\"tabs\" aria-label=\"{html.escape(game.name)}视图\">{buttons}</nav>"


def _render_game_overview(game: GameDashboard) -> str:
    return f"<section class=\"view active\" data-view=\"overview-{html.escape(game.code)}\"><div class=\"game-grid\">{_render_game_card(game)}</div></section>"


def _render_game_candidates(game: GameDashboard) -> str:
    content = _render_candidate_groups(game) if game.candidates else "<p class=\"empty\">暂无候选号码报告。</p>"
    return f"<section class=\"view\" data-view=\"candidates-{html.escape(game.code)}\">{content}</section>"


def _render_game_analysis(game: GameDashboard) -> str:
    analysis = game.analysis
    if analysis.draw_count <= 0:
        content = "<p class=\"empty\">暂无本地开奖数据，先更新该彩种开奖数据后再查看分析。</p>"
    else:
        sections = "".join(_render_analysis_section(section) for section in analysis.sections)
        omissions = "".join(_render_analysis_section(section) for section in analysis.omission_sections)
        content = f"""
<section class="panel">
  <h2>{html.escape(game.name)} 历史号码分析</h2>
  <p class="panel-note">基于本地 normalized CSV 统计，只描述历史出现频率，不代表未来概率。</p>
  <dl class="analysis-metrics">
    <div><dt>统计期数</dt><dd>{analysis.draw_count}</dd></div>
    <div><dt>最新期号</dt><dd>{html.escape(analysis.latest_issue or '-')}</dd></div>
    <div><dt>最新日期</dt><dd>{html.escape(analysis.latest_draw_date or '-')}</dd></div>
  </dl>
  <div class="analysis-grid">{sections}</div>
  <h3 class="analysis-subtitle">遗漏排行</h3>
  <div class="analysis-grid">{omissions}</div>
  {_render_sum_ranges(analysis.sum_ranges)}
  {_render_partition_trends(analysis.partition_trends)}
  {_render_candidate_score_table(analysis.candidate_scores)}
</section>
""".strip()
    return f"<section class=\"view\" data-view=\"analysis-{html.escape(game.code)}\">{content}</section>"


def _render_analysis_section(section: DashboardAnalysisSection) -> str:
    if not section.frequencies:
        rows = "<p class=\"empty\">暂无可统计号码。</p>"
    else:
        max_count = max(item.count for item in section.frequencies) or 1
        rows = "\n".join(
            _render_frequency_item(item, max_count, "期" if "遗漏" in section.label else "次")
            for item in section.frequencies
        )
    return f"""
<section class="analysis-section">
  <h3>{html.escape(section.label)}</h3>
  <div class="frequency-list">{rows}</div>
</section>
""".strip()


def _render_frequency_item(item: DashboardNumberFrequency, max_count: int, unit: str = "次") -> str:
    width = max(6, min(100, round(item.count / max_count * 100)))
    return f"""
<div class="frequency-item">
  <span class="number-pill">{html.escape(item.number)}</span>
  <span class="frequency-bar"><i style="width:{width}%"></i></span>
  <span class="frequency-count">{item.count}{html.escape(unit)}</span>
</div>
""".strip()


def _render_sum_ranges(buckets: Sequence[DashboardRangeBucket]) -> str:
    if not buckets:
        return ""
    rows = "\n".join(
        f"<tr><td>{html.escape(bucket.label)}</td><td>{bucket.count}</td><td>{bucket.rate:.1%}</td></tr>"
        for bucket in buckets
    )
    return f"""
<section class="analysis-section analysis-wide">
  <h3>和值区间</h3>
  <table><thead><tr><th>区间</th><th>期数</th><th>占比</th></tr></thead><tbody>{rows}</tbody></table>
</section>
""".strip()


def _render_partition_trends(trends: Sequence[DashboardPartitionTrend]) -> str:
    if not trends:
        return ""
    rows = "\n".join(
        f"<tr><td>{html.escape(item.label)}</td><td>{item.average:.2f}</td></tr>"
        for item in trends
    )
    return f"""
<section class="analysis-section analysis-wide">
  <h3>分区走势</h3>
  <p class="panel-note">按低区/中区/高区统计平均每期出现数量，用来看号码分布是否偏向某一区。</p>
  <table><thead><tr><th>分区</th><th>平均每期数量</th></tr></thead><tbody>{rows}</tbody></table>
</section>
""".strip()


def _render_candidate_score_table(scores: Sequence[DashboardCandidateScore]) -> str:
    if not scores:
        return ""
    rows = "\n".join(
        "<tr>"
        f"<td>{item.rank}</td>"
        f"<td>{html.escape(item.strategy)}</td>"
        f"<td>{html.escape(item.number)}</td>"
        f"<td>{item.score}</td>"
        f"<td>{html.escape(item.explanation)}</td>"
        "</tr>"
        for item in scores
    )
    return f"""
<section class="analysis-section analysis-wide">
  <h3>候选号码评分解释</h3>
  <p class="panel-note">评分只解释候选号码和历史统计的贴合程度，不代表真实中奖概率。</p>
  <table><thead><tr><th>排名</th><th>策略</th><th>号码</th><th>评分</th><th>解释</th></tr></thead><tbody>{rows}</tbody></table>
</section>
""".strip()


def _render_candidate_groups(game: GameDashboard) -> str:
    primary = _primary_candidates(game)
    reference = [candidate for candidate in game.candidates if candidate not in primary and "随机" not in candidate.strategy]
    sections = [
        _render_candidate_table(game, "主推荐", primary[:3], "优先查看这一组，其他策略只作为参考。"),
        _render_candidate_table(game, "参考候选", reference, "热号、冷号、遗漏和机器学习候选用于对比。随机基线已从候选展示中隐藏。"),
    ]
    return "".join(sections)


def _primary_candidates(game: GameDashboard) -> List[DashboardCandidate]:
    ml_candidates = [candidate for candidate in game.candidates if "机器学习" in candidate.strategy]
    if ml_candidates:
        return ml_candidates
    non_random = [candidate for candidate in game.candidates if "随机" not in candidate.strategy]
    return non_random[:1] or list(game.candidates[:1])


def _render_game_summary(game: GameDashboard, summary_text: str) -> str:
    scoped = _filter_summary_for_game(summary_text, game)
    note = "<p class=\"panel-note\">已开奖后才计算投入和奖金；待开奖记录只增加推荐记录数，不会增加已开奖投入。</p>"
    content = note + (_pre(scoped) if scoped else "<p class=\"empty\">暂无该彩种长期汇总。</p>")
    content += _render_recommendation_history_table(game)
    return f"<section class=\"view\" data-view=\"summary-{html.escape(game.code)}\"><section class=\"panel\"><h2>{html.escape(game.name)} 长期汇总</h2>{content}</section></section>"


def _render_recommendation_history_table(game: GameDashboard) -> str:
    if not game.recent_recommendations:
        return "<section class=\"recommendation-history\"><h3>历史推荐记录</h3><p class=\"empty\">暂无已保存推荐记录。</p></section>"
    rows = "\n".join(
        _render_recommendation_history_row(record, game.target_draw_dates.get(record.target_issue, ""))
        for record in game.recent_recommendations
    )
    return f"""
<section class="recommendation-history">
  <h3>历史推荐记录</h3>
  <p class="panel-note">同一期多次生成会按不同批次保留；开奖数据更新后，待开奖记录会被校验并写入命中结果。</p>
  <table>
    <thead><tr><th>目标期</th><th>预计开奖日期</th><th>批次</th><th>生成时间</th><th>排名</th><th>策略</th><th>号码</th><th>开奖号码</th><th>奖级/命中</th><th>命中明细</th><th>奖金</th><th>状态</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</section>
""".strip()


def _render_recommendation_history_row(record: RecommendationRecord, draw_date: str = "") -> str:
    status = "已开奖" if record.status == "checked" else "待开奖"
    prize = record.prize_level or ("待开奖" if record.status == "pending" else "未中奖")
    hit_detail = _render_hit_detail(record)
    return (
        "<tr>"
        f"<td>{html.escape(record.target_issue or '-')}</td>"
        f"<td>{html.escape(draw_date or '-')}</td>"
        f"<td>{html.escape(record.run_id or '-')}</td>"
        f"<td>{html.escape(record.generated_at or '-')}</td>"
        f"<td>{html.escape(str(record.rank))}</td>"
        f"<td>{html.escape(record.strategy_name)}</td>"
        f"<td>{html.escape(record.numbers)}</td>"
        f"<td>{html.escape(record.draw_numbers or '-')}</td>"
        f"<td>{html.escape(prize)}</td>"
        f"<td>{hit_detail}</td>"
        f"<td>{_format_dashboard_amount(record.payout)}</td>"
        f"<td>{status}</td>"
        "</tr>"
    )


def _render_hit_detail(record: RecommendationRecord) -> str:
    detail = _recommendation_hit_detail(record)
    if not detail:
        return "-"
    summary, hit_text, miss_text = detail
    return (
        "<div class=\"hit-detail\">"
        "<strong>命中明细</strong>"
        f"<span>{html.escape(summary)}</span>"
        f"<span>命中：{html.escape(hit_text)}</span>"
        f"<span>未中：{html.escape(miss_text)}</span>"
        "</div>"
    )


def _recommendation_hit_detail(record: RecommendationRecord) -> Optional[Tuple[str, str, str]]:
    if record.status != "checked" or not record.draw_numbers:
        return None

    if record.game_code == "qlc":
        return _qlc_hit_detail(record)

    prediction_sections = _split_number_sections(record.game_code, record.numbers)
    draw_sections = _split_number_sections(record.game_code, record.draw_numbers)
    if not prediction_sections or not draw_sections:
        return None

    labels = HIT_SECTION_LABELS.get(record.game_code, ("号码",))
    summaries: List[str] = []
    hit_sections: List[str] = []
    miss_sections: List[str] = []
    for index, predicted in enumerate(prediction_sections):
        actual = draw_sections[index] if index < len(draw_sections) else ()
        actual_set = set(actual)
        hits = tuple(number for number in predicted if number in actual_set)
        misses = tuple(number for number in predicted if number not in actual_set)
        label = labels[index] if index < len(labels) else "号码"
        summaries.append(f"{label} {len(hits)}/{len(predicted)}")
        hit_sections.append(" ".join(hits))
        miss_sections.append(" ".join(misses))

    return (
        "，".join(summaries),
        _join_number_sections(hit_sections) or "无",
        _join_number_sections(miss_sections) or "无",
    )


def _qlc_hit_detail(record: RecommendationRecord) -> Optional[Tuple[str, str, str]]:
    predicted = _split_plain_numbers(record.numbers)
    draw_sections = _split_number_sections(record.game_code, record.draw_numbers)
    if not predicted or not draw_sections:
        return None

    basic = draw_sections[0] if draw_sections else ()
    special = draw_sections[1][0] if len(draw_sections) > 1 and draw_sections[1] else ""
    basic_set = set(basic)
    basic_hits = tuple(number for number in predicted if number in basic_set)
    special_hits = (special,) if special and special in predicted else ()
    misses = tuple(number for number in predicted if number not in basic_set and number != special)
    return (
        f"基本号 {len(basic_hits)}/{len(predicted)}，特别号 {len(special_hits)}/1",
        _join_number_sections((" ".join(basic_hits), " ".join(special_hits))) or "无",
        " ".join(misses) or "无",
    )


def _split_number_sections(game_code: str, value: str) -> Tuple[Tuple[str, ...], ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    if "+" in text:
        return tuple(_split_plain_numbers(part) for part in text.split("+"))
    return (_split_plain_numbers(text, compact_digits=game_code in {"3d", "pl3", "pl5"}),)


def _split_plain_numbers(value: str, compact_digits: bool = False) -> Tuple[str, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    if compact_digits and " " not in text:
        return tuple(ch for ch in text if ch.isdigit())
    return tuple(part.strip() for part in re.split(r"\s+", text) if part.strip())


def _join_number_sections(sections: Sequence[str]) -> str:
    return " + ".join(section for section in sections if section)


def _render_game_reports(game: GameDashboard) -> str:
    blocks = [
        _render_report_block(f"{game.name} {label}", text)
        for label, text in game.reports.items()
        if text
    ]
    if not blocks:
        blocks.append("<p class=\"empty\">暂无该彩种文本报告。</p>")
    return f"<section class=\"view\" data-view=\"reports-{html.escape(game.code)}\">{''.join(blocks)}</section>"


def _default_game_index(games: Sequence[GameDashboard]) -> int:
    for index, game in enumerate(games):
        if game.code == "ssq":
            return index
    return 0


def _render_game_card(game: GameDashboard) -> str:
    candidate = game.candidates[0].number if game.candidates else "-"
    latest_target = game.latest_target_issue or "-"
    latest_prize = game.latest_prize_level or "-"
    payout = _format_dashboard_amount(game.recommendation_payout)
    return f"""
<article class="game-card">
  <div class="game-card-head">
    <h2>{html.escape(game.name)}</h2>
    <span>{html.escape(game.code.upper())}</span>
  </div>
  <div class="metric-row">
    <div><dt>推荐总数</dt><dd>{game.recommendation_total}</dd></div>
    <div><dt>已开奖</dt><dd>{game.recommendation_checked}</dd></div>
    <div><dt>待开奖</dt><dd>{game.recommendation_pending}</dd></div>
    <div><dt>中奖记录</dt><dd>{game.recommendation_winning}</dd></div>
    <div><dt>奖金</dt><dd>{html.escape(payout)}</dd></div>
    <div><dt>最新目标期</dt><dd>{html.escape(latest_target)}</dd></div>
  </div>
  <p class="metric-note">总览只统计已保存的推荐记录；开奖数据更新后会校验待开奖记录并填入中奖结果。</p>
  <div class="candidate-strip">
    <small>最新目标状态</small>
    <strong>{html.escape(latest_prize)}</strong>
  </div>
  <div class="candidate-strip">
    <small>当前首位候选</small>
    <strong>{html.escape(candidate)}</strong>
  </div>
</article>
""".strip()


def _render_candidate_table(
    game: GameDashboard,
    title: Optional[str] = None,
    candidates: Optional[Sequence[DashboardCandidate]] = None,
    note: str = "",
) -> str:
    rows_source = tuple(candidates if candidates is not None else game.candidates)
    if not rows_source:
        return f"""
<section class="panel">
  <h2>{html.escape(title or (game.name + " 候选号码"))}</h2>
  <p class="empty">{html.escape(note or "暂无候选号码。")}</p>
</section>
""".strip()
    rows = "\n".join(
        "<tr>"
        f"<td>{candidate.rank}</td>"
        f"<td>{html.escape(candidate.strategy)}</td>"
        f"<td><strong>{html.escape(candidate.number)}</strong></td>"
        "</tr>"
        for candidate in rows_source
    )
    note_html = f"<p class=\"panel-note\">{html.escape(note)}</p>" if note else ""
    return f"""
<section class="panel">
  <h2>{html.escape(title or (game.name + " 候选号码"))}</h2>
  {note_html}
  <table>
    <thead><tr><th>排名</th><th>策略</th><th>号码</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</section>
""".strip()


def _render_report_block(title: str, text: str) -> str:
    return f"<section class=\"panel\"><h2>{html.escape(title)}</h2>{_pre(text)}</section>"


def _filter_summary_for_game(summary_text: str, game: GameDashboard) -> str:
    if not summary_text:
        return ""
    lines = summary_text.splitlines()
    table_start = next((index for index, line in enumerate(lines) if line.startswith("彩种 |")), -1)
    if table_start < 0:
        names = SUMMARY_GAME_ALIASES.get(game.code, (game.name,))
        return summary_text if any(name in summary_text for name in names) else ""
    scoped = ["推荐长期表现汇总", "", lines[table_start]]
    if table_start + 1 < len(lines):
        scoped.append(lines[table_start + 1])
    for line in lines[table_start + 2:]:
        first_cell = line.split("|", 1)[0].strip()
        if _summary_row_belongs_to_game(first_cell, game):
            scoped.append(line)
    return "\n".join(scoped) if len(scoped) > 4 else ""


def _summary_row_belongs_to_game(first_cell: str, game: GameDashboard) -> bool:
    aliases = SUMMARY_GAME_ALIASES.get(game.code, (game.name,))
    return any(first_cell == alias or first_cell.startswith(alias) for alias in aliases)


def _parse_metric(text: str, label: str) -> str:
    match = re.search(rf"{re.escape(label)}：([^\n]+)", text)
    if not match:
        return ""
    return match.group(1).strip()


def _parse_candidates(text: str) -> List[DashboardCandidate]:
    candidates: List[DashboardCandidate] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        candidates.append(DashboardCandidate(
            rank=int(parts[0]),
            strategy=parts[1],
            number=parts[2],
        ))
    return candidates


def _merge_candidates(*candidate_groups: Sequence[DashboardCandidate]) -> List[DashboardCandidate]:
    merged: List[DashboardCandidate] = []
    seen = set()
    for group in candidate_groups:
        for candidate in group:
            key = (candidate.strategy, candidate.number)
            if key in seen:
                continue
            seen.add(key)
            merged.append(candidate)
    return merged


def _resolve_history_data_dir(reports_path: Path) -> Path:
    reports_path = Path(reports_path)
    if reports_path.name == "latest" and reports_path.parent.name == "reports":
        return reports_path.parent.parent / "data" / "normalized"
    return Path("data/normalized")


def _strip_money(value: str) -> str:
    return value.replace("元", "").strip()


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _pre(text: str) -> str:
    return f"<pre>{html.escape(text)}</pre>"


def _make_dashboard_handler(reports_dir: Path, repo_root: Path):
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                model = load_dashboard_model(reports_dir)
                self._send(200, "text/html; charset=utf-8", render_dashboard_html(model).encode("utf-8"))
                return
            if path == "/health":
                body = json.dumps({"ok": True}, ensure_ascii=False).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body)
                return
            if path == "/api/export":
                params = parse_qs(urlparse(self.path).query)
                export_format = (params.get("format") or ["csv"])[0]
                try:
                    model = load_dashboard_model(reports_dir)
                    path_out = export_dashboard_snapshot(model, Path(repo_root) / "reports" / "exports", export_format)
                    body = json.dumps({
                        "ok": True,
                        "path": str(path_out),
                    }, ensure_ascii=False).encode("utf-8")
                    self._send(200, "application/json; charset=utf-8", body)
                except ValueError as exc:
                    body = json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False).encode("utf-8")
                    self._send(400, "application/json; charset=utf-8", body)
                return
            if path.startswith("/api/jobs/"):
                parts = [part for part in path.split("/") if part]
                job_id = parts[2] if len(parts) >= 3 else ""
                job = _get_dashboard_job(job_id)
                if job is None:
                    self._send(404, "application/json; charset=utf-8", json.dumps({
                        "ok": False,
                        "error": "job not found",
                    }, ensure_ascii=False).encode("utf-8"))
                    return
                if len(parts) == 4 and parts[3] == "events":
                    self._send_job_events(job)
                    return
                body = json.dumps(_job_snapshot(job), ensure_ascii=False).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body)
                return
            self._send(404, "text/plain; charset=utf-8", "Not Found".encode("utf-8"))

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/api/config":
                length = int(self.headers.get("Content-Length", "0") or 0)
                raw = self.rfile.read(length).decode("utf-8") if length > 0 else ""
                params = parse_qs(raw)
                save_dashboard_config(
                    _dashboard_db_path_for_root(repo_root),
                    {key: values[0] for key, values in params.items() if values},
                )
                body = json.dumps({"ok": True}, ensure_ascii=False).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body)
                return
            if path.startswith("/api/jobs/"):
                action = path.rsplit("/", 1)[-1]
                params = parse_qs(parsed.query)
                game_code = (params.get("game") or [""])[0]
                options = {
                    key: values[0]
                    for key, values in params.items()
                    if key != "game" and values
                }
                job = start_dashboard_job(action, repo_root, game_code=game_code, options=options)
                status_code = 202 if job.status == "running" else 400
                body = json.dumps(_job_snapshot(job), ensure_ascii=False).encode("utf-8")
                self._send(status_code, "application/json; charset=utf-8", body)
                return
            if path.startswith("/api/actions/"):
                action = path.rsplit("/", 1)[-1]
                params = parse_qs(parsed.query)
                game_code = (params.get("game") or [""])[0]
                options = {
                    key: values[0]
                    for key, values in params.items()
                    if key != "game" and values
                }
                result = run_dashboard_action(action, repo_root, game_code=game_code, options=options)
                body = json.dumps({
                    "action": result.action,
                    "ok": result.ok,
                    "exit_code": result.exit_code,
                    "command": result.command,
                    "output": result.output,
                }, ensure_ascii=False).encode("utf-8")
                self._send(200 if result.ok else 500, "application/json; charset=utf-8", body)
                return
            self._send(404, "text/plain; charset=utf-8", "Not Found".encode("utf-8"))

        def log_message(self, format, *args) -> None:  # noqa: A002
            return

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _send_job_events(self, job: DashboardJob) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            for event in _job_sse_events(job):
                self.wfile.write(event.encode("utf-8"))
                self.wfile.flush()

    return DashboardHandler


def _dashboard_css() -> str:
    return """
:root {
  --bg: #f7f8fa;
  --surface: #ffffff;
  --surface-2: #f2f4f7;
  --text: #1f2937;
  --muted: #667085;
  --line: #e4e7ec;
  --blue: #2563eb;
  --green: #087443;
  --amber: #b54708;
  --rose: #b42318;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
  line-height: 1.5;
}
.shell {
  width: min(1280px, 100%);
  margin: 0 auto;
  padding: 20px;
}
.app-header {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: center;
  padding: 14px 0;
  border-bottom: 1px solid var(--line);
}
h1, h2, p { margin: 0; }
h1 { font-size: 22px; font-weight: 700; }
h2 { font-size: 18px; font-weight: 700; }
.app-header p, .empty { color: var(--muted); margin-top: 4px; }
.header-metrics {
  display: grid;
  grid-template-columns: repeat(3, minmax(88px, auto));
  gap: 8px;
  margin: 0;
}
.header-metrics div, .metric-row div {
  border: 1px solid var(--line);
  background: var(--surface);
  border-radius: 6px;
  padding: 8px 10px;
}
dt { color: var(--muted); font-size: 12px; }
dd { margin: 0; font-size: 16px; font-weight: 700; }
.risk {
  margin: 12px 0;
  padding: 9px 10px;
  border: 1px solid #fecdca;
  border-radius: 6px;
  background: #fffbfa;
  color: #7a271a;
}
.actions {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 14px;
  margin: 12px 0;
}
.section-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
  margin-bottom: 10px;
}
.section-head span {
  color: var(--muted);
  font-size: 13px;
}
.action-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
}
.action-head span {
  color: var(--muted);
  font-size: 13px;
}
#action-status {
  color: var(--muted);
  font-weight: 700;
}
#action-status.running { color: var(--amber); }
#action-status.ok { color: var(--green); }
#action-status.error { color: var(--rose); }
.action-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}
.game-action-tabs {
  display: flex;
  gap: 6px;
  overflow-x: auto;
  margin-top: 12px;
  border-bottom: 1px solid var(--line);
}
.game-action-tab {
  appearance: none;
  border: 0;
  background: transparent;
  color: var(--muted);
  cursor: pointer;
  padding: 9px 10px;
  font: inherit;
  border-bottom: 3px solid transparent;
  white-space: nowrap;
}
.game-action-tab.active {
  color: var(--blue);
  border-color: var(--blue);
  font-weight: 700;
}
.game-action-pane { display: none; }
.game-action-pane.active { display: block; }
.param-panel {
  margin-top: 12px;
  border: 1px solid var(--line);
  background: #fbfcfe;
  border-radius: 8px;
  padding: 12px;
}
.param-title {
  font-weight: 700;
  margin-bottom: 4px;
}
.param-panel p, .panel-note {
  color: var(--muted);
  margin: 4px 0 0;
  font-size: 13px;
}
.ml-context {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 8px;
  margin: 10px 0 0;
}
.ml-context div {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
  padding: 8px 10px;
}
.ml-context dd {
  font-size: 16px;
}
.basic-param-grid,
.param-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 10px;
  margin-top: 10px;
}
.param-field {
  display: grid;
  gap: 5px;
  color: var(--text);
  font-size: 13px;
}
.param-label {
  display: flex;
  align-items: center;
  gap: 5px;
  min-width: 0;
  font-weight: 700;
}
.param-field input {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 8px 9px;
  font: inherit;
}
.param-desc {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.4;
}
.advanced-params {
  margin-top: 10px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
  padding: 8px 10px;
}
.advanced-params summary {
  color: var(--muted);
  cursor: pointer;
  font-weight: 700;
}
.lottery-tabs {
  display: flex;
  gap: 6px;
  overflow-x: auto;
  margin: 18px 0 10px;
  border-bottom: 1px solid var(--line);
}
.lottery-tab {
  appearance: none;
  border: 0;
  background: transparent;
  color: var(--muted);
  cursor: pointer;
  padding: 10px 12px;
  font: inherit;
  border-bottom: 3px solid transparent;
  white-space: nowrap;
}
.lottery-tab.active {
  color: var(--green);
  border-color: var(--green);
  font-weight: 700;
}
.lottery-pane { display: none; }
.lottery-pane.active { display: block; }
.action-button {
  appearance: none;
  border: 1px solid var(--line);
  background: var(--surface-2);
  color: var(--text);
  cursor: pointer;
  border-radius: 8px;
  padding: 9px 12px;
  font: inherit;
  font-weight: 700;
}
.action-button.primary {
  background: var(--blue);
  border-color: var(--blue);
  color: #fff;
}
.action-button:disabled {
  cursor: wait;
  opacity: 0.65;
}
#action-output {
  min-height: 82px;
}
.action-log {
  margin-top: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface-2);
  padding: 10px;
}
.action-log-head {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  align-items: center;
}
.action-log-actions {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  justify-content: flex-end;
}
.action-clear {
  appearance: none;
  border: 1px solid var(--line);
  background: var(--surface);
  border-radius: 8px;
  cursor: pointer;
  padding: 5px 9px;
  font: inherit;
}
#action-summary {
  margin-top: 6px;
  color: var(--text);
}
.progress-panel {
  margin-top: 10px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
  padding: 10px;
}
.progress-head,
.progress-meta {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  color: var(--muted);
  font-size: 13px;
}
.progress-head span:first-child {
  color: var(--text);
  font-weight: 700;
}
#action-progress {
  width: 100%;
  height: 12px;
  margin: 8px 0;
  accent-color: var(--green);
}
#action-details {
  margin-top: 8px;
}
.recommendation-history-global {
  margin-top: 12px;
}
.recommendation-history-summary {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
  cursor: pointer;
  list-style: none;
}
.recommendation-history-summary::-webkit-details-marker {
  display: none;
}
.recommendation-history-summary strong {
  font-size: 18px;
}
.history-toggle {
  border: 1px solid #bfdbfe;
  border-radius: 999px;
  color: var(--blue);
  background: #eff6ff;
  padding: 4px 10px;
  font-size: 13px;
  white-space: nowrap;
}
.history-toggle-open {
  display: none;
}
.recommendation-history-global[open] .history-toggle-open {
  display: inline-block;
}
.recommendation-history-global[open] .history-toggle-closed {
  display: none;
}
.recommendation-batches {
  display: grid;
  gap: 8px;
  margin-top: 10px;
}
.backend-panel {
  margin-bottom: 14px;
}
.backend-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
  margin-top: 12px;
}
.backend-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  background: #fff;
}
.backend-card h3 {
  margin: 0 0 10px;
  font-size: 15px;
}
.backend-card label {
  display: grid;
  gap: 5px;
  margin-bottom: 8px;
  color: var(--muted);
  font-size: 13px;
}
.backend-card input {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 8px 10px;
  color: var(--text);
}
.export-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.recommendation-batch {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
  overflow: hidden;
}
.recommendation-batch summary {
  display: grid;
  grid-template-columns: 1.1fr 0.9fr 0.9fr 1.2fr repeat(5, auto);
  gap: 8px;
  align-items: center;
  padding: 10px 12px;
  cursor: pointer;
  list-style: none;
}
.recommendation-batch summary::-webkit-details-marker {
  display: none;
}
.recommendation-batch summary span {
  color: var(--muted);
  font-size: 13px;
}
.recommendation-batch summary span:first-child {
  color: var(--text);
  font-weight: 700;
}
.recommendation-batch summary strong,
.batch-toggle-text {
  color: var(--blue);
  font-size: 13px;
  white-space: nowrap;
}
.batch-toggle-text {
  border: 1px solid #bfdbfe;
  border-radius: 999px;
  padding: 4px 8px;
  background: #eff6ff;
  text-align: center;
}
.batch-detail {
  border-top: 1px solid var(--line);
  padding: 10px 12px 12px;
  overflow-x: auto;
}
.batch-id {
  color: var(--muted);
  font-size: 13px;
}
.tabs {
  display: flex;
  gap: 6px;
  margin: 18px 0;
  border-bottom: 1px solid var(--line);
}
.tab {
  appearance: none;
  border: 0;
  background: transparent;
  color: var(--muted);
  cursor: pointer;
  padding: 10px 12px;
  font: inherit;
  border-bottom: 3px solid transparent;
}
.tab.active {
  color: var(--blue);
  border-color: var(--blue);
  font-weight: 700;
}
.view { display: none; }
.view.active { display: block; }
.game-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px;
}
.game-card, .panel {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
}
.game-card-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
  margin-bottom: 14px;
}
.game-card-head span {
  color: var(--muted);
  font-weight: 700;
}
.metric-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(118px, 1fr));
  gap: 8px;
}
.metric-note {
  color: var(--muted);
  font-size: 12px;
  margin-top: 8px;
}
.candidate-strip {
  margin-top: 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0 0;
  border-top: 1px solid var(--line);
}
.candidate-strip small { color: var(--muted); }
.candidate-strip strong { font-size: 22px; color: var(--green); }
.panel { margin-bottom: 12px; }
.analysis-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 8px;
  margin: 12px 0;
}
.analysis-metrics div {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfcfe;
  padding: 8px 10px;
}
.analysis-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px;
}
.analysis-subtitle {
  margin: 18px 0 10px;
}
.analysis-section {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  background: var(--surface);
}
.analysis-wide {
  margin-top: 12px;
}
.analysis-section h3 {
  margin: 0 0 10px;
  font-size: 15px;
}
.frequency-list {
  display: grid;
  gap: 8px;
}
.frequency-item {
  display: grid;
  grid-template-columns: 42px minmax(90px, 1fr) 48px;
  gap: 8px;
  align-items: center;
  font-size: 13px;
}
.number-pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 34px;
  border-radius: 999px;
  border: 1px solid #bfdbfe;
  background: #eff6ff;
  color: var(--blue);
  font-weight: 700;
  padding: 3px 7px;
}
.frequency-bar {
  height: 8px;
  border-radius: 999px;
  background: var(--surface-2);
  overflow: hidden;
}
.frequency-bar i {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: var(--green);
}
.frequency-count {
  color: var(--muted);
  text-align: right;
  white-space: nowrap;
}
.hit-detail {
  display: grid;
  gap: 2px;
  min-width: 150px;
  color: var(--muted);
  font-size: 12px;
}
.hit-detail strong {
  color: var(--text);
  font-size: 12px;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
}
th, td {
  border-bottom: 1px solid var(--line);
  padding: 9px 8px;
  text-align: left;
  vertical-align: top;
}
th { color: var(--muted); font-size: 13px; }
pre {
  margin: 10px 0 0;
  overflow: auto;
  white-space: pre-wrap;
  background: #101828;
  color: #eef4ff;
  border-radius: 8px;
  padding: 14px;
  font-family: Consolas, "Microsoft YaHei", monospace;
  font-size: 13px;
}
@media (max-width: 760px) {
  .shell { padding: 14px; }
  .app-header { display: block; }
  .header-metrics { grid-template-columns: 1fr; margin-top: 12px; }
  .section-head { display: block; }
  .tabs { overflow-x: auto; }
  .metric-row { grid-template-columns: 1fr; }
  .action-head { display: block; }
  .action-buttons { display: grid; }
  .action-button { width: 100%; }
  .progress-head, .progress-meta { display: grid; gap: 4px; }
  .recommendation-batch summary { grid-template-columns: 1fr; }
  .basic-param-grid, .param-grid { grid-template-columns: 1fr; }
  .param-label { align-items: flex-start; justify-content: space-between; }
  .candidate-strip { align-items: flex-start; flex-direction: column; gap: 4px; }
  .candidate-strip strong { font-size: 18px; overflow-wrap: anywhere; }
  .panel { overflow-x: auto; }
  table { min-width: 560px; }
  pre { font-size: 12px; }
}
""".strip()


def _dashboard_js() -> str:
    return "\n".join([
        """
const tabs = document.querySelectorAll('.tab');
tabs.forEach((tab) => {
  tab.addEventListener('click', () => {
    const target = tab.dataset.target;
    const scope = tab.closest('[data-lottery-pane]') || document;
    scope.querySelectorAll('.tab').forEach((item) => item.classList.toggle('active', item === tab));
    scope.querySelectorAll('.view').forEach((view) => view.classList.toggle('active', view.dataset.view === target));
  });
});
const lotteryTabs = document.querySelectorAll('[data-lottery-tab]');
const lotteryPanes = document.querySelectorAll('[data-lottery-pane]');
lotteryTabs.forEach((tab) => {
  tab.addEventListener('click', () => {
    const game = tab.dataset.lotteryTab;
    lotteryTabs.forEach((item) => item.classList.toggle('active', item === tab));
    lotteryPanes.forEach((pane) => pane.classList.toggle('active', pane.dataset.lotteryPane === game));
  });
});
const actionButtons = document.querySelectorAll('[data-action]');
const actionLog = document.getElementById('action-log');
const actionStatus = document.getElementById('action-status');
const actionSummary = document.getElementById('action-summary');
const actionOutput = document.getElementById('action-output');
const actionDetails = document.getElementById('action-details');
const clearActionOutput = document.getElementById('clear-action-output');
const refreshDashboardData = document.getElementById('refresh-dashboard-data');
const progressPanel = document.getElementById('progress-panel');
const progressStage = document.getElementById('progress-stage');
const actionProgress = document.getElementById('action-progress');
const progressElapsed = document.getElementById('progress-elapsed');
const progressRemaining = document.getElementById('progress-remaining');
const gameActionTabs = document.querySelectorAll('[data-game-tab]');
const gameActionPanes = document.querySelectorAll('[data-game-pane]');
const recommendationBatches = document.querySelectorAll('.recommendation-batch');
const llmConfigForm = document.getElementById('llm-config-form');
const configSaveStatus = document.getElementById('config-save-status');
const exportStatus = document.getElementById('export-status');
let actionInFlight = false;
function setActionState(label, className) {
  actionStatus.textContent = label;
  actionStatus.className = className;
}
function actionParams(button) {
  const params = new URLSearchParams();
  params.set('game', button.dataset.game);
  const pane = button.closest('[data-game-pane]');
  if (pane) {
    pane.querySelectorAll('[data-action-param]').forEach((input) => {
      if (input.value !== '') {
        params.set(input.dataset.actionParam, input.value);
      }
    });
  }
  return params;
}
gameActionTabs.forEach((tab) => {
  tab.addEventListener('click', () => {
    const game = tab.dataset.gameTab;
    gameActionTabs.forEach((item) => item.classList.toggle('active', item === tab));
    gameActionPanes.forEach((pane) => pane.classList.toggle('active', pane.dataset.gamePane === game));
  });
});
clearActionOutput.addEventListener('click', () => {
  actionLog.hidden = true;
  actionOutput.textContent = '';
  actionSummary.textContent = '';
  actionDetails.open = false;
  progressPanel.hidden = true;
  refreshDashboardData.hidden = true;
});
function refreshDashboardPage() {
  const nextUrl = new URL(window.location.href);
  nextUrl.searchParams.set('_ts', String(Date.now()));
  window.location.replace(nextUrl.toString());
}
refreshDashboardData.addEventListener('click', refreshDashboardPage);
recommendationBatches.forEach((batch) => {
  const label = batch.querySelector('.batch-toggle-text');
  if (!label) {
    return;
  }
  const updateLabel = () => {
    label.textContent = batch.open ? '收起明细' : '展开明细';
  };
  updateLabel();
  batch.addEventListener('toggle', updateLabel);
});
if (llmConfigForm) {
  llmConfigForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const response = await fetch('/api/config', {
      method: 'POST',
      body: new URLSearchParams(new FormData(llmConfigForm)),
    });
    configSaveStatus.textContent = response.ok ? '配置已保存到本地 SQLite' : '配置保存失败';
  });
}
document.querySelectorAll('[data-export]').forEach((button) => {
  button.addEventListener('click', async () => {
    const format = button.dataset.export;
    const response = await fetch(`/api/export?format=${format}`);
    const result = await response.json();
    exportStatus.textContent = result.ok ? `已导出：${result.path}` : `导出失败：${result.error || ''}`;
  });
});
function formatSeconds(value) {
  if (value === null || value === undefined) {
    return '计算中';
  }
  const seconds = Math.max(0, Number(value) || 0);
  if (seconds < 60) {
    return `${seconds}秒`;
  }
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return `${minutes}分${rest}秒`;
}
function renderJobProgress(job) {
  progressPanel.hidden = false;
  progressStage.textContent = `${job.stage_label || '执行中'} (${job.stage_index || 0}/${job.stage_total || 1})`;
  actionProgress.value = job.progress_percent || 0;
  progressElapsed.textContent = `已用时间：${formatSeconds(job.elapsed_seconds)}`;
  progressRemaining.textContent = `预计剩余：${formatSeconds(job.estimated_remaining_seconds)}`;
  actionOutput.textContent = [
    `$ ${job.command || ''}`,
    `exit_code: ${job.exit_code === null || job.exit_code === undefined ? '运行中' : job.exit_code}`,
    job.output || '(no output)',
  ].join('\\n');
}
async function pollActionJob(jobId, button) {
  while (true) {
    const response = await fetch(`/api/jobs/${jobId}`);
    const job = await response.json();
    renderJobProgress(job);
    if (job.status === 'completed' || job.status === 'failed') {
      setActionState(job.status === 'completed' ? '完成' : '失败', job.status === 'completed' ? 'ok' : 'error');
      actionSummary.textContent = job.status === 'completed' ? `${button.textContent}完成` : `${button.textContent}失败`;
      actionDetails.open = job.status !== 'completed';
      refreshDashboardAfterJob(job);
      return job;
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}
function watchActionJob(jobId, button) {
  if (!window.EventSource) {
    return pollActionJob(jobId, button);
  }
  return new Promise((resolve, reject) => {
    const source = new EventSource(`/api/jobs/${jobId}/events`);
    source.onmessage = (event) => {
      const job = JSON.parse(event.data);
      renderJobProgress(job);
      if (job.status === 'completed' || job.status === 'failed') {
        source.close();
        setActionState(job.status === 'completed' ? '瀹屾垚' : '澶辫触', job.status === 'completed' ? 'ok' : 'error');
        actionSummary.textContent = job.status === 'completed' ? `${button.textContent}瀹屾垚` : `${button.textContent}澶辫触`;
        actionDetails.open = job.status !== 'completed';
        refreshDashboardAfterJob(job);
        resolve(job);
      }
    };
    source.onerror = () => {
      source.close();
      pollActionJob(jobId, button).then(resolve).catch(reject);
    };
  });
}
function refreshDashboardAfterJob(job) {
  if (job.status !== 'completed') {
    return;
  }
  actionSummary.textContent = `${actionSummary.textContent}，数据已写入，点“刷新页面数据”查看最新总览和推荐历史`;
  refreshDashboardData.hidden = false;
}
actionButtons.forEach((button) => {
  button.addEventListener('click', async () => {
    if (actionInFlight) {
      return;
    }
    actionInFlight = true;
    const action = button.dataset.action;
    const params = actionParams(button);
    actionButtons.forEach((item) => { item.disabled = true; });
    actionLog.hidden = false;
    actionDetails.open = true;
    progressPanel.hidden = false;
    actionProgress.value = 1;
    progressStage.textContent = '准备执行';
    progressElapsed.textContent = '已用时间：0秒';
    progressRemaining.textContent = '预计剩余：计算中';
    setActionState('执行中', 'running');
    actionSummary.textContent = `${button.textContent} ...`;
    actionOutput.textContent = actionSummary.textContent;
    refreshDashboardData.hidden = true;
    try {
      const response = await fetch(`/api/jobs/${action}?${params.toString()}`, { method: 'POST' });
      const job = await response.json();
      renderJobProgress(job);
      if (!response.ok && job.status !== 'running') {
        throw new Error(job.output || '任务启动失败');
      }
      await watchActionJob(job.job_id, button);
    } catch (error) {
      setActionState('失败', 'error');
      actionOutput.textContent = String(error);
    } finally {
      actionInFlight = false;
      actionButtons.forEach((item) => { item.disabled = false; });
    }
  });
});
""",
    ]).strip()
