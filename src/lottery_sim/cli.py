import argparse
from pathlib import Path
from typing import Optional, Sequence

from lottery_sim.analysis.stability import (
    SegmentMetric,
    summarize_result_by_issue_year,
    summarize_seed_sensitivity,
)
from lottery_sim.analysis.metrics import winning_bet_count
from lottery_sim.backtest.engine import run_backtest
from lottery_sim.backtest.dlt_engine import run_dlt_backtest
from lottery_sim.backtest.kl8_engine import run_kl8_backtest
from lottery_sim.backtest.qlc_engine import run_qlc_backtest
from lottery_sim.backtest.qxc_engine import run_qxc_backtest
from lottery_sim.backtest.ssq_engine import run_ssq_backtest
from lottery_sim.data_sources.dlt_17500 import (
    fetch_17500_dlt_text,
    load_dlt_draws_csv,
    parse_17500_dlt_text,
    save_dlt_draws_csv,
)
from lottery_sim.data_sources.fucai3d_17500 import (
    fetch_17500_3d_text,
    load_draws_csv,
    parse_17500_3d_text,
    save_draws_csv,
)
from lottery_sim.data_sources.kl8_17500 import (
    fetch_17500_kl8_text,
    load_kl8_draws_csv,
    parse_17500_kl8_text,
    save_kl8_draws_csv,
)
from lottery_sim.data_sources.pl3_17500 import (
    fetch_17500_pl3_text,
    parse_17500_pl3_text,
)
from lottery_sim.data_sources.pl5_17500 import (
    fetch_17500_pl5_text,
    load_pl5_draws_csv,
    parse_17500_pl5_text,
    save_pl5_draws_csv,
)
from lottery_sim.data_sources.qxc_17500 import (
    fetch_17500_qxc_text,
    load_qxc_draws_csv,
    parse_17500_qxc_text,
    save_qxc_draws_csv,
)
from lottery_sim.data_sources.qlc_17500 import (
    fetch_17500_qlc_text,
    load_qlc_draws_csv,
    parse_17500_qlc_text,
    save_qlc_draws_csv,
)
from lottery_sim.data_sources.ssq_17500 import (
    fetch_17500_ssq_text,
    load_ssq_draws_csv,
    parse_17500_ssq_text,
    save_ssq_draws_csv,
)
from lottery_sim.data_sources.incremental import render_incremental_update_result, update_draws_csv
from lottery_sim.dashboard import serve_dashboard
from lottery_sim.fastapi_app import serve_fastapi_dashboard
from lottery_sim.games.fucai3d import Fucai3DGame
from lottery_sim.games.dlt import DltGame
from lottery_sim.games.kl8 import Kl8Game
from lottery_sim.games.pl3 import PL3Game
from lottery_sim.games.pl5 import PL5Game
from lottery_sim.games.qlc import QlcGame
from lottery_sim.games.qxc import QxcGame
from lottery_sim.games.ssq import SsqGame
from lottery_sim.issue_calendar import next_issue_from_latest_draw
from lottery_sim.ml.ssq import (
    load_ssq_ml_model,
    recommend_ssq_ml,
    run_ssq_ml_backtest,
    save_ssq_ml_model,
    train_ssq_ml_model,
)
from lottery_sim.ml.generic import (
    load_generic_ml_model,
    ml_adapter_for_game,
    recommend_generic_ml,
    run_generic_ml_backtest,
    save_generic_ml_model,
    train_generic_ml_model,
)
from lottery_sim.recommendation_store import RecommendationStore
from lottery_sim.recommendation_tracking import (
    available_recommendation_draws,
    create_recommendation_records,
    evaluate_recommendation_records,
    load_recommendation_records,
    parse_pick_text,
    render_recommendation_verification_report,
    save_recommendation_records,
    select_recommendation_window,
)
from lottery_sim.recommendation_summary import (
    load_records_from_paths,
    render_recommendation_summary_report,
    summarize_recommendation_records,
)
from lottery_sim.recommendations import generate_candidates, render_recommendation_report
from lottery_sim.reports.compare_report import render_compare_report
from lottery_sim.reports.stability_report import render_stability_report
from lottery_sim.reports.text_report import render_backtest_report
from lottery_sim.strategies.random_5d import Random5DStrategy
from lottery_sim.strategies.random_strategy import Random3DStrategy
from lottery_sim.strategies.random_dlt import RandomDltStrategy
from lottery_sim.strategies.random_kl8 import RandomKl8Strategy
from lottery_sim.strategies.random_qlc import RandomQlcStrategy
from lottery_sim.strategies.random_qxc import RandomQxcStrategy
from lottery_sim.strategies.random_ssq import RandomSsqStrategy
from lottery_sim.strategies.statistical_3d import (
    Cold3DStrategy,
    Hot3DStrategy,
    Omission3DStrategy,
    SumNearest3DStrategy,
)
from lottery_sim.strategies.statistical_5d import (
    Cold5DStrategy,
    Hot5DStrategy,
    Omission5DStrategy,
)
from lottery_sim.strategies.statistical_dlt import (
    ColdDltStrategy,
    HotDltStrategy,
    OmissionDltStrategy,
)
from lottery_sim.strategies.statistical_kl8 import (
    ColdKl8Strategy,
    HotKl8Strategy,
    OmissionKl8Strategy,
)
from lottery_sim.strategies.statistical_qxc import (
    ColdQxcStrategy,
    HotQxcStrategy,
    OmissionQxcStrategy,
)
from lottery_sim.strategies.statistical_qlc import (
    ColdQlcStrategy,
    HotQlcStrategy,
    OmissionQlcStrategy,
)
from lottery_sim.strategies.statistical_ssq import (
    ColdSsqStrategy,
    HotSsqStrategy,
    OmissionSsqStrategy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lottery-sim",
        description="彩票模拟分析命令行工具",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize = subparsers.add_parser("normalize-3d", help="解析并标准化17500福彩3D历史数据")
    normalize.add_argument("--input", help="本地17500 TXT文件路径。不传则从默认URL抓取")
    normalize.add_argument("--url", help="17500 TXT URL，不传则使用默认正序地址")
    normalize.add_argument("--output", required=True, help="输出标准CSV路径")
    normalize.set_defaults(func=_normalize_3d)

    normalize_pl3 = subparsers.add_parser("normalize-pl3", help="解析并标准化17500排列三历史数据")
    normalize_pl3.add_argument("--input", help="本地17500 TXT文件路径。不传则从默认URL抓取")
    normalize_pl3.add_argument("--url", help="17500 TXT URL，不传则使用默认正序地址")
    normalize_pl3.add_argument("--output", required=True, help="输出标准CSV路径")
    normalize_pl3.set_defaults(func=_normalize_pl3)

    normalize_pl5 = subparsers.add_parser("normalize-pl5", help="解析并标准化17500排列五历史数据")
    normalize_pl5.add_argument("--input", help="本地17500 TXT文件路径。不传则从默认URL抓取")
    normalize_pl5.add_argument("--url", help="17500 TXT URL，不传则使用默认正序地址")
    normalize_pl5.add_argument("--output", required=True, help="输出标准CSV路径")
    normalize_pl5.set_defaults(func=_normalize_pl5)

    normalize_qxc = subparsers.add_parser("normalize-qxc", help="解析并标准化17500七星彩历史数据")
    normalize_qxc.add_argument("--input", help="本地17500 TXT文件路径。不传则从默认URL抓取")
    normalize_qxc.add_argument("--url", help="17500 TXT URL，不传则使用默认正序地址")
    normalize_qxc.add_argument("--output", required=True, help="输出标准CSV路径")
    normalize_qxc.set_defaults(func=_normalize_qxc)

    normalize_qlc = subparsers.add_parser("normalize-qlc", help="解析并标准化17500七乐彩历史数据")
    normalize_qlc.add_argument("--input", help="本地17500 TXT文件路径。不传则从默认URL抓取")
    normalize_qlc.add_argument("--url", help="17500 TXT URL，不传则使用默认正序地址")
    normalize_qlc.add_argument("--output", required=True, help="输出标准CSV路径")
    normalize_qlc.set_defaults(func=_normalize_qlc)

    normalize_kl8 = subparsers.add_parser("normalize-kl8", help="解析并标准化17500快乐8历史数据")
    normalize_kl8.add_argument("--input", help="本地17500 TXT文件路径。不传则从默认URL抓取")
    normalize_kl8.add_argument("--url", help="17500 TXT URL，不传则使用默认正序地址")
    normalize_kl8.add_argument("--output", required=True, help="输出标准CSV路径")
    normalize_kl8.set_defaults(func=_normalize_kl8)

    normalize_ssq = subparsers.add_parser("normalize-ssq", help="解析并标准化17500双色球历史数据")
    normalize_ssq.add_argument("--input", help="本地17500 TXT文件路径。不传则从默认URL抓取")
    normalize_ssq.add_argument("--url", help="17500 TXT URL，不传则使用默认正序地址")
    normalize_ssq.add_argument("--output", required=True, help="输出标准CSV路径")
    normalize_ssq.set_defaults(func=_normalize_ssq)

    normalize_dlt = subparsers.add_parser("normalize-dlt", help="解析并标准化17500大乐透历史数据")
    normalize_dlt.add_argument("--input", help="本地17500 TXT文件路径。不传则从默认URL抓取")
    normalize_dlt.add_argument("--url", help="17500 TXT URL，不传则使用默认正序地址")
    normalize_dlt.add_argument("--output", required=True, help="输出标准CSV路径")
    normalize_dlt.set_defaults(func=_normalize_dlt)

    _add_update_parser(subparsers, "update-3d", "增量更新福彩3D开奖数据", _update_3d)
    _add_update_parser(subparsers, "update-pl3", "增量更新排列三开奖数据", _update_pl3)
    _add_update_parser(subparsers, "update-pl5", "增量更新排列五开奖数据", _update_pl5)
    _add_update_parser(subparsers, "update-qxc", "增量更新7星彩开奖数据", _update_qxc)
    _add_update_parser(subparsers, "update-qlc", "增量更新七乐彩开奖数据", _update_qlc)
    _add_update_parser(subparsers, "update-kl8", "增量更新快乐8开奖数据", _update_kl8)
    _add_update_parser(subparsers, "update-ssq", "增量更新双色球开奖数据", _update_ssq)
    _add_update_parser(subparsers, "update-dlt", "增量更新大乐透开奖数据", _update_dlt)

    backtest = subparsers.add_parser("backtest-3d", help="对标准CSV执行福彩3D随机基线回测")
    backtest.add_argument("--csv", required=True, help="标准CSV路径")
    backtest.add_argument("--seed", type=int, default=20260505, help="随机种子")
    backtest.add_argument("--min-history", type=int, default=1, help="最少历史期数")
    backtest.set_defaults(func=_backtest_3d)

    backtest_pl3 = subparsers.add_parser("backtest-pl3", help="对标准CSV执行排列三随机基线回测")
    backtest_pl3.add_argument("--csv", required=True, help="标准CSV路径")
    backtest_pl3.add_argument("--seed", type=int, default=20260505, help="随机种子")
    backtest_pl3.add_argument("--min-history", type=int, default=1, help="最少历史期数")
    backtest_pl3.set_defaults(func=_backtest_pl3)

    backtest_pl5 = subparsers.add_parser("backtest-pl5", help="对标准CSV执行排列五随机基线回测")
    backtest_pl5.add_argument("--csv", required=True, help="标准CSV路径")
    backtest_pl5.add_argument("--seed", type=int, default=20260505, help="随机种子")
    backtest_pl5.add_argument("--min-history", type=int, default=1, help="最少历史期数")
    backtest_pl5.set_defaults(func=_backtest_pl5)

    backtest_qxc = subparsers.add_parser("backtest-qxc", help="对标准CSV执行七星彩随机基线回测")
    backtest_qxc.add_argument("--csv", required=True, help="标准CSV路径")
    backtest_qxc.add_argument("--seed", type=int, default=20260505, help="随机种子")
    backtest_qxc.add_argument("--min-history", type=int, default=1, help="最少历史期数")
    backtest_qxc.set_defaults(func=_backtest_qxc)

    backtest_qlc = subparsers.add_parser("backtest-qlc", help="对标准CSV执行七乐彩随机基线回测")
    backtest_qlc.add_argument("--csv", required=True, help="标准CSV路径")
    backtest_qlc.add_argument("--seed", type=int, default=20260505, help="随机种子")
    backtest_qlc.add_argument("--min-history", type=int, default=1, help="最少历史期数")
    backtest_qlc.set_defaults(func=_backtest_qlc)

    backtest_kl8 = subparsers.add_parser("backtest-kl8", help="对标准CSV执行快乐8随机基线回测")
    backtest_kl8.add_argument("--csv", required=True, help="标准CSV路径")
    backtest_kl8.add_argument("--pick-size", type=int, default=10, help="任选玩法号码个数，1到10")
    backtest_kl8.add_argument("--seed", type=int, default=20260505, help="随机种子")
    backtest_kl8.add_argument("--min-history", type=int, default=1, help="最少历史期数")
    backtest_kl8.set_defaults(func=_backtest_kl8)

    backtest_ssq = subparsers.add_parser("backtest-ssq", help="对标准CSV执行双色球随机基线回测")
    backtest_ssq.add_argument("--csv", required=True, help="标准CSV路径")
    backtest_ssq.add_argument("--seed", type=int, default=20260505, help="随机种子")
    backtest_ssq.add_argument("--min-history", type=int, default=1, help="最少历史期数")
    backtest_ssq.set_defaults(func=_backtest_ssq)

    backtest_dlt = subparsers.add_parser("backtest-dlt", help="对标准CSV执行大乐透随机基线回测")
    backtest_dlt.add_argument("--csv", required=True, help="标准CSV路径")
    backtest_dlt.add_argument("--seed", type=int, default=20260505, help="随机种子")
    backtest_dlt.add_argument("--min-history", type=int, default=1, help="最少历史期数")
    backtest_dlt.set_defaults(func=_backtest_dlt)

    compare = subparsers.add_parser("compare-3d", help="对福彩3D执行多策略回测对比")
    compare.add_argument("--csv", required=True, help="标准CSV路径")
    compare.add_argument("--seed", type=int, default=20260505, help="随机基线种子")
    compare.add_argument("--window", type=int, default=30, help="统计策略窗口期数")
    compare.add_argument("--min-history", type=int, default=30, help="最少历史期数")
    compare.set_defaults(func=_compare_3d)

    compare_pl3 = subparsers.add_parser("compare-pl3", help="对排列三执行多策略回测对比")
    compare_pl3.add_argument("--csv", required=True, help="标准CSV路径")
    compare_pl3.add_argument("--seed", type=int, default=20260505, help="随机基线种子")
    compare_pl3.add_argument("--window", type=int, default=30, help="统计策略窗口期数")
    compare_pl3.add_argument("--min-history", type=int, default=30, help="最少历史期数")
    compare_pl3.set_defaults(func=_compare_pl3)

    compare_pl5 = subparsers.add_parser("compare-pl5", help="对排列五执行多策略回测对比")
    compare_pl5.add_argument("--csv", required=True, help="标准CSV路径")
    compare_pl5.add_argument("--seed", type=int, default=20260505, help="随机基线种子")
    compare_pl5.add_argument("--window", type=int, default=30, help="统计策略窗口期数")
    compare_pl5.add_argument("--min-history", type=int, default=30, help="最少历史期数")
    compare_pl5.set_defaults(func=_compare_pl5)

    compare_qxc = subparsers.add_parser("compare-qxc", help="对七星彩执行多策略回测对比")
    compare_qxc.add_argument("--csv", required=True, help="标准CSV路径")
    compare_qxc.add_argument("--seed", type=int, default=20260505, help="随机基线种子")
    compare_qxc.add_argument("--window", type=int, default=30, help="统计策略窗口期数")
    compare_qxc.add_argument("--min-history", type=int, default=30, help="最少历史期数")
    compare_qxc.set_defaults(func=_compare_qxc)

    compare_qlc = subparsers.add_parser("compare-qlc", help="对七乐彩执行多策略回测对比")
    compare_qlc.add_argument("--csv", required=True, help="标准CSV路径")
    compare_qlc.add_argument("--seed", type=int, default=20260505, help="随机基线种子")
    compare_qlc.add_argument("--window", type=int, default=30, help="统计策略窗口期数")
    compare_qlc.add_argument("--min-history", type=int, default=30, help="最少历史期数")
    compare_qlc.set_defaults(func=_compare_qlc)

    compare_kl8 = subparsers.add_parser("compare-kl8", help="对快乐8执行多策略回测对比")
    compare_kl8.add_argument("--csv", required=True, help="标准CSV路径")
    compare_kl8.add_argument("--pick-size", type=int, default=10, help="任选玩法号码个数，1到10")
    compare_kl8.add_argument("--seed", type=int, default=20260505, help="随机基线种子")
    compare_kl8.add_argument("--window", type=int, default=30, help="统计策略窗口期数")
    compare_kl8.add_argument("--min-history", type=int, default=30, help="最少历史期数")
    compare_kl8.set_defaults(func=_compare_kl8)

    compare_ssq = subparsers.add_parser("compare-ssq", help="对双色球执行多策略回测对比")
    compare_ssq.add_argument("--csv", required=True, help="标准CSV路径")
    compare_ssq.add_argument("--seed", type=int, default=20260505, help="随机基线种子")
    compare_ssq.add_argument("--window", type=int, default=30, help="统计策略窗口期数")
    compare_ssq.add_argument("--min-history", type=int, default=30, help="最少历史期数")
    compare_ssq.set_defaults(func=_compare_ssq)

    compare_dlt = subparsers.add_parser("compare-dlt", help="对大乐透执行多策略回测对比")
    compare_dlt.add_argument("--csv", required=True, help="标准CSV路径")
    compare_dlt.add_argument("--seed", type=int, default=20260505, help="随机基线种子")
    compare_dlt.add_argument("--window", type=int, default=30, help="统计策略窗口期数")
    compare_dlt.add_argument("--min-history", type=int, default=30, help="最少历史期数")
    compare_dlt.set_defaults(func=_compare_dlt)

    stability = subparsers.add_parser("stability-3d", help="评估福彩3D策略分年稳定性")
    stability.add_argument("--csv", required=True, help="标准CSV路径")
    stability.add_argument("--seeds", default="1,2,3,4,5", help="随机基线种子列表，逗号分隔")
    stability.add_argument("--window", type=int, default=30, help="统计策略窗口期数")
    stability.add_argument("--min-history", type=int, default=30, help="最少历史期数")
    stability.set_defaults(func=_stability_3d)

    stability_pl3 = subparsers.add_parser("stability-pl3", help="评估排列三策略分年稳定性")
    stability_pl3.add_argument("--csv", required=True, help="标准CSV路径")
    stability_pl3.add_argument("--seeds", default="1,2,3,4,5", help="随机基线种子列表，逗号分隔")
    stability_pl3.add_argument("--window", type=int, default=30, help="统计策略窗口期数")
    stability_pl3.add_argument("--min-history", type=int, default=30, help="最少历史期数")
    stability_pl3.set_defaults(func=_stability_pl3)

    stability_pl5 = subparsers.add_parser("stability-pl5", help="评估排列五策略分年稳定性")
    stability_pl5.add_argument("--csv", required=True, help="标准CSV路径")
    stability_pl5.add_argument("--seeds", default="1,2,3,4,5", help="随机基线种子列表，逗号分隔")
    stability_pl5.add_argument("--window", type=int, default=30, help="统计策略窗口期数")
    stability_pl5.add_argument("--min-history", type=int, default=30, help="最少历史期数")
    stability_pl5.set_defaults(func=_stability_pl5)

    stability_qxc = subparsers.add_parser("stability-qxc", help="评估七星彩策略分年稳定性")
    stability_qxc.add_argument("--csv", required=True, help="标准CSV路径")
    stability_qxc.add_argument("--seeds", default="1,2,3,4,5", help="随机基线种子列表，逗号分隔")
    stability_qxc.add_argument("--window", type=int, default=30, help="统计策略窗口期数")
    stability_qxc.add_argument("--min-history", type=int, default=30, help="最少历史期数")
    stability_qxc.set_defaults(func=_stability_qxc)

    stability_qlc = subparsers.add_parser("stability-qlc", help="评估七乐彩策略分年稳定性")
    stability_qlc.add_argument("--csv", required=True, help="标准CSV路径")
    stability_qlc.add_argument("--seeds", default="1,2,3,4,5", help="随机基线种子列表，逗号分隔")
    stability_qlc.add_argument("--window", type=int, default=30, help="统计策略窗口期数")
    stability_qlc.add_argument("--min-history", type=int, default=30, help="最少历史期数")
    stability_qlc.set_defaults(func=_stability_qlc)

    stability_kl8 = subparsers.add_parser("stability-kl8", help="评估快乐8策略分年稳定性")
    stability_kl8.add_argument("--csv", required=True, help="标准CSV路径")
    stability_kl8.add_argument("--pick-size", type=int, default=10, help="任选玩法号码个数，1到10")
    stability_kl8.add_argument("--seeds", default="1,2,3,4,5", help="随机基线种子列表，逗号分隔")
    stability_kl8.add_argument("--window", type=int, default=30, help="统计策略窗口期数")
    stability_kl8.add_argument("--min-history", type=int, default=30, help="最少历史期数")
    stability_kl8.set_defaults(func=_stability_kl8)

    stability_ssq = subparsers.add_parser("stability-ssq", help="评估双色球策略分年稳定性")
    stability_ssq.add_argument("--csv", required=True, help="标准CSV路径")
    stability_ssq.add_argument("--seeds", default="1,2,3,4,5", help="随机基线种子列表，逗号分隔")
    stability_ssq.add_argument("--window", type=int, default=30, help="统计策略窗口期数")
    stability_ssq.add_argument("--min-history", type=int, default=30, help="最少历史期数")
    stability_ssq.set_defaults(func=_stability_ssq)

    stability_dlt = subparsers.add_parser("stability-dlt", help="评估大乐透策略分年稳定性")
    stability_dlt.add_argument("--csv", required=True, help="标准CSV路径")
    stability_dlt.add_argument("--seeds", default="1,2,3,4,5", help="随机基线种子列表，逗号分隔")
    stability_dlt.add_argument("--window", type=int, default=30, help="统计策略窗口期数")
    stability_dlt.add_argument("--min-history", type=int, default=30, help="最少历史期数")
    stability_dlt.set_defaults(func=_stability_dlt)

    _add_recommend_parser(subparsers, "recommend-3d", "生成福彩3D候选号码", _recommend_3d)
    _add_recommend_parser(subparsers, "recommend-pl3", "生成排列三候选号码", _recommend_pl3)
    _add_recommend_parser(subparsers, "recommend-pl5", "生成排列五候选号码", _recommend_pl5)
    _add_recommend_parser(subparsers, "recommend-qxc", "生成7星彩候选号码", _recommend_qxc)
    _add_recommend_parser(subparsers, "recommend-qlc", "生成七乐彩候选号码", _recommend_qlc)
    _add_recommend_parser(subparsers, "recommend-kl8", "生成快乐8候选号码", _recommend_kl8, pick_size=True)
    _add_recommend_parser(subparsers, "recommend-ssq", "生成双色球候选号码", _recommend_ssq)
    _add_recommend_parser(subparsers, "recommend-dlt", "生成大乐透候选号码", _recommend_dlt)

    train_ml_ssq = subparsers.add_parser("train-ml-ssq", help="训练双色球机器学习模型")
    train_ml_ssq.add_argument("--csv", required=True, help="标准CSV路径")
    train_ml_ssq.add_argument("--model", default="models/ssq-ml.json", help="模型JSON输出路径")
    train_ml_ssq.add_argument("--min-history", type=int, default=30, help="生成训练样本所需最少历史期数")
    train_ml_ssq.add_argument("--epochs", type=int, default=30, help="训练轮数")
    train_ml_ssq.add_argument("--learning-rate", type=float, default=0.04, help="学习率")
    train_ml_ssq.set_defaults(func=_train_ml_ssq)

    backtest_ml_ssq = subparsers.add_parser("backtest-ml-ssq", help="滚动回测双色球机器学习模型")
    backtest_ml_ssq.add_argument("--csv", required=True, help="标准CSV路径")
    backtest_ml_ssq.add_argument("--min-train", type=int, default=200, help="首次训练所需历史期数")
    backtest_ml_ssq.add_argument("--min-history", type=int, default=30, help="生成训练样本所需最少历史期数")
    backtest_ml_ssq.add_argument("--limit", type=int, default=120, help="最近多少期参与滚动回测")
    backtest_ml_ssq.add_argument("--retrain-every", type=int, default=20, help="每隔多少期重训一次模型")
    backtest_ml_ssq.add_argument("--epochs", type=int, default=12, help="每次训练轮数")
    backtest_ml_ssq.set_defaults(func=_backtest_ml_ssq)

    recommend_ml_ssq = subparsers.add_parser("recommend-ml-ssq", help="生成双色球机器学习候选号码")
    recommend_ml_ssq.add_argument("--csv", required=True, help="标准CSV路径")
    recommend_ml_ssq.add_argument("--model", default="models/ssq-ml.json", help="模型JSON路径，不存在时自动训练")
    recommend_ml_ssq.add_argument("--count", type=int, default=10, help="候选号码数量")
    recommend_ml_ssq.add_argument("--min-history", type=int, default=30, help="自动训练时的最少历史期数")
    recommend_ml_ssq.add_argument("--epochs", type=int, default=30, help="自动训练时的训练轮数")
    recommend_ml_ssq.set_defaults(func=_recommend_ml_ssq)

    record_ml_ssq = subparsers.add_parser("record-recommend-ml-ssq", help="保存双色球机器学习推荐记录")
    record_ml_ssq.add_argument("--store", default="data/recommendations", help="recommendation record store root")
    record_ml_ssq.add_argument("--csv", required=True, help="标准CSV路径")
    record_ml_ssq.add_argument("--model", default="models/ssq-ml.json", help="模型JSON路径，不存在时自动训练")
    record_ml_ssq.add_argument("--output", default="", help="推荐记录CSV输出路径；不传则写入 --store")
    record_ml_ssq.add_argument("--target-issue", default="", help="目标开奖期号。不传则按开奖日历规则推算")
    record_ml_ssq.add_argument("--history-until", default="", help="推荐生成只使用该期号及之前的历史数据")
    record_ml_ssq.add_argument("--count", type=int, default=10, help="候选号码数量")
    record_ml_ssq.add_argument("--min-history", type=int, default=30, help="自动训练时的最少历史期数")
    record_ml_ssq.add_argument("--epochs", type=int, default=30, help="自动训练时的训练轮数")
    record_ml_ssq.set_defaults(func=_record_recommend_ml_ssq)

    _add_generic_ml_parsers(subparsers, "3d", "福彩3D", _train_ml_3d, _backtest_ml_3d, _recommend_ml_3d, _record_recommend_ml_3d)
    _add_generic_ml_parsers(subparsers, "pl3", "排列三", _train_ml_pl3, _backtest_ml_pl3, _recommend_ml_pl3, _record_recommend_ml_pl3)
    _add_generic_ml_parsers(subparsers, "pl5", "排列五", _train_ml_pl5, _backtest_ml_pl5, _recommend_ml_pl5, _record_recommend_ml_pl5)
    _add_generic_ml_parsers(subparsers, "qxc", "7星彩", _train_ml_qxc, _backtest_ml_qxc, _recommend_ml_qxc, _record_recommend_ml_qxc)
    _add_generic_ml_parsers(subparsers, "qlc", "七乐彩", _train_ml_qlc, _backtest_ml_qlc, _recommend_ml_qlc, _record_recommend_ml_qlc)
    _add_generic_ml_parsers(subparsers, "kl8", "快乐8", _train_ml_kl8, _backtest_ml_kl8, _recommend_ml_kl8, _record_recommend_ml_kl8, pick_size=True)
    _add_generic_ml_parsers(subparsers, "dlt", "大乐透", _train_ml_dlt, _backtest_ml_dlt, _recommend_ml_dlt, _record_recommend_ml_dlt)

    _add_record_recommend_parser(subparsers, "record-recommend-3d", "保存福彩3D推荐记录", _record_recommend_3d)
    _add_record_recommend_parser(subparsers, "record-recommend-pl3", "保存排列三推荐记录", _record_recommend_pl3)
    _add_record_recommend_parser(subparsers, "record-recommend-pl5", "保存排列五推荐记录", _record_recommend_pl5)
    _add_record_recommend_parser(subparsers, "record-recommend-qxc", "保存7星彩推荐记录", _record_recommend_qxc)
    _add_record_recommend_parser(subparsers, "record-recommend-qlc", "保存七乐彩推荐记录", _record_recommend_qlc)
    _add_record_recommend_parser(subparsers, "record-recommend-kl8", "保存快乐8推荐记录", _record_recommend_kl8, pick_size=True)
    _add_record_recommend_parser(subparsers, "record-recommend-ssq", "保存双色球推荐记录", _record_recommend_ssq)
    _add_record_recommend_parser(subparsers, "record-recommend-dlt", "保存大乐透推荐记录", _record_recommend_dlt)

    _add_verify_recommend_parser(subparsers, "verify-recommend-3d", "校验福彩3D推荐记录", _verify_recommend_3d)
    _add_verify_recommend_parser(subparsers, "verify-recommend-pl3", "校验排列三推荐记录", _verify_recommend_pl3)
    _add_verify_recommend_parser(subparsers, "verify-recommend-pl5", "校验排列五推荐记录", _verify_recommend_pl5)
    _add_verify_recommend_parser(subparsers, "verify-recommend-qxc", "校验7星彩推荐记录", _verify_recommend_qxc)
    _add_verify_recommend_parser(subparsers, "verify-recommend-qlc", "校验七乐彩推荐记录", _verify_recommend_qlc)
    _add_verify_recommend_parser(subparsers, "verify-recommend-kl8", "校验快乐8推荐记录", _verify_recommend_kl8, pick_size=True)
    _add_verify_recommend_parser(subparsers, "verify-recommend-ssq", "校验双色球推荐记录", _verify_recommend_ssq)
    _add_verify_recommend_parser(subparsers, "verify-recommend-dlt", "校验大乐透推荐记录", _verify_recommend_dlt)

    summary = subparsers.add_parser("summarize-recommendations", help="汇总多期推荐记录的长期表现")
    summary.add_argument("--records", nargs="+", required=True, help="一个或多个推荐记录CSV路径")
    summary.add_argument("--output", help="汇总报告输出路径。不传则输出到终端")
    summary.set_defaults(func=_summarize_recommendations)

    dashboard = subparsers.add_parser("dashboard", help="启动本地Web仪表盘")
    dashboard.add_argument("--reports", default="reports/latest", help="报告目录")
    dashboard.add_argument("--host", default="127.0.0.1", help="监听地址")
    dashboard.add_argument("--port", type=int, default=8765, help="监听端口")
    dashboard.add_argument("--server", choices=("auto", "stdlib", "fastapi"), default="auto", help="Web服务类型")
    dashboard.add_argument("--open", dest="open_browser", action="store_true", help="启动后自动打开浏览器")
    dashboard.add_argument("--no-open", dest="open_browser", action="store_false", help="启动后不自动打开浏览器")
    dashboard.set_defaults(func=_dashboard, open_browser=False)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


def _add_recommend_parser(subparsers, command: str, help_text: str, func, pick_size: bool = False) -> None:
    parser = subparsers.add_parser(command, help=help_text)
    parser.add_argument("--csv", required=True, help="标准CSV路径")
    parser.add_argument("--count", type=int, default=10, help="候选号码数量")
    parser.add_argument("--seed", type=int, default=20260505, help="随机补充种子")
    parser.add_argument("--window", type=int, default=30, help="统计策略窗口期数")
    if pick_size:
        parser.add_argument("--pick-size", type=int, default=10, help="任选玩法号码个数，1到10")
    parser.set_defaults(func=func)


def _add_update_parser(subparsers, command: str, help_text: str, func) -> None:
    parser = subparsers.add_parser(command, help=help_text)
    parser.add_argument("--csv", required=True, help="本地标准CSV路径")
    parser.add_argument("--input", help="本地17500 TXT文件路径。不传则从默认URL抓取")
    parser.add_argument("--url", help="17500 TXT URL，不传则使用默认正序地址")
    parser.set_defaults(func=func)


def _add_record_recommend_parser(subparsers, command: str, help_text: str, func, pick_size: bool = False) -> None:
    parser = subparsers.add_parser(command, help=help_text)
    parser.add_argument("--store", default="data/recommendations", help="recommendation record store root")
    parser.add_argument("--csv", required=True, help="标准CSV路径")
    parser.add_argument("--output", default="", help="推荐记录CSV输出路径；不传则写入 --store")
    parser.add_argument("--target-issue", default="", help="目标开奖期号。不传则按开奖日历规则推算")
    parser.add_argument("--history-until", default="", help="推荐生成只使用该期号及之前的历史数据")
    parser.add_argument("--count", type=int, default=10, help="候选号码数量")
    parser.add_argument("--seed", type=int, default=20260505, help="随机补充种子")
    parser.add_argument("--window", type=int, default=30, help="统计策略窗口期数")
    if pick_size:
        parser.add_argument("--pick-size", type=int, default=10, help="任选玩法号码个数，1到10")
    parser.set_defaults(func=func)


def _add_generic_ml_parsers(
    subparsers,
    code: str,
    label: str,
    train_func,
    backtest_func,
    recommend_func,
    record_func,
    pick_size: bool = False,
) -> None:
    train = subparsers.add_parser(f"train-ml-{code}", help=f"训练{label}机器学习模型")
    train.add_argument("--csv", required=True, help="标准CSV路径")
    train.add_argument("--model", default=f"models/{code}-ml.json", help="模型JSON输出路径")
    train.add_argument("--min-history", type=int, default=30, help="生成训练样本所需最少历史期数")
    train.add_argument("--epochs", type=int, default=30, help="训练轮数")
    train.add_argument("--learning-rate", type=float, default=0.04, help="学习率")
    if pick_size:
        train.add_argument("--pick-size", type=int, default=10, help="快乐8任选玩法号码个数，1到10")
    train.set_defaults(func=train_func)

    backtest = subparsers.add_parser(f"backtest-ml-{code}", help=f"滚动回测{label}机器学习模型")
    backtest.add_argument("--csv", required=True, help="标准CSV路径")
    backtest.add_argument("--min-train", type=int, default=200, help="首次训练所需历史期数")
    backtest.add_argument("--min-history", type=int, default=30, help="生成训练样本所需最少历史期数")
    backtest.add_argument("--limit", type=int, default=120, help="最近多少期参与滚动回测")
    backtest.add_argument("--retrain-every", type=int, default=20, help="每隔多少期重训一次模型")
    backtest.add_argument("--epochs", type=int, default=12, help="每次训练轮数")
    if pick_size:
        backtest.add_argument("--pick-size", type=int, default=10, help="快乐8任选玩法号码个数，1到10")
    backtest.set_defaults(func=backtest_func)

    recommend = subparsers.add_parser(f"recommend-ml-{code}", help=f"生成{label}机器学习候选号码")
    recommend.add_argument("--csv", required=True, help="标准CSV路径")
    recommend.add_argument("--model", default=f"models/{code}-ml.json", help="模型JSON路径，不存在时自动训练")
    recommend.add_argument("--count", type=int, default=10, help="候选号码数量")
    recommend.add_argument("--min-history", type=int, default=30, help="自动训练时的最少历史期数")
    recommend.add_argument("--epochs", type=int, default=30, help="自动训练时的训练轮数")
    if pick_size:
        recommend.add_argument("--pick-size", type=int, default=10, help="快乐8任选玩法号码个数，1到10")
    recommend.set_defaults(func=recommend_func)

    record = subparsers.add_parser(f"record-recommend-ml-{code}", help=f"保存{label}机器学习推荐记录")
    record.add_argument("--store", default="data/recommendations", help="recommendation record store root")
    record.add_argument("--csv", required=True, help="标准CSV路径")
    record.add_argument("--model", default=f"models/{code}-ml.json", help="模型JSON路径，不存在时自动训练")
    record.add_argument("--output", default="", help="推荐记录CSV输出路径；不传则写入 --store")
    record.add_argument("--target-issue", default="", help="目标开奖期号。不传则按开奖日历规则推算")
    record.add_argument("--history-until", default="", help="推荐生成只使用该期号及之前的历史数据")
    record.add_argument("--count", type=int, default=10, help="候选号码数量")
    record.add_argument("--min-history", type=int, default=30, help="自动训练时的最少历史期数")
    record.add_argument("--epochs", type=int, default=30, help="自动训练时的训练轮数")
    if pick_size:
        record.add_argument("--pick-size", type=int, default=10, help="快乐8任选玩法号码个数，1到10")
    record.set_defaults(func=record_func)


def _add_verify_recommend_parser(subparsers, command: str, help_text: str, func, pick_size: bool = False) -> None:
    parser = subparsers.add_parser(command, help=help_text)
    parser.add_argument("--csv", required=True, help="标准CSV路径")
    parser.add_argument("--records", required=True, help="推荐记录CSV路径")
    parser.add_argument("--output", help="校验后推荐记录CSV输出路径")
    if pick_size:
        parser.add_argument("--pick-size", type=int, default=10, help="任选玩法号码个数，1到10")
    parser.set_defaults(func=func)


def _normalize_3d(args) -> None:
    _normalize_pick3(
        args=args,
        fetch_text=fetch_17500_3d_text,
        parse_text=parse_17500_3d_text,
        label="福彩3D",
    )


def _normalize_pl3(args) -> None:
    _normalize_pick3(
        args=args,
        fetch_text=fetch_17500_pl3_text,
        parse_text=parse_17500_pl3_text,
        label="排列三",
    )


def _normalize_pl5(args) -> None:
    if args.input:
        text = _read_text_file(Path(args.input))
    else:
        text = fetch_17500_pl5_text(url=args.url) if args.url else fetch_17500_pl5_text()

    draws = parse_17500_pl5_text(text)
    save_pl5_draws_csv(draws, Path(args.output))
    print(f"已保存 {len(draws)} 条排列五开奖数据：{args.output}")


def _normalize_qxc(args) -> None:
    if args.input:
        text = _read_text_file(Path(args.input))
    else:
        text = fetch_17500_qxc_text(url=args.url) if args.url else fetch_17500_qxc_text()

    draws = parse_17500_qxc_text(text)
    save_qxc_draws_csv(draws, Path(args.output))
    print(f"已保存 {len(draws)} 条7星彩开奖数据：{args.output}")


def _normalize_qlc(args) -> None:
    if args.input:
        text = _read_text_file(Path(args.input))
    else:
        text = fetch_17500_qlc_text(url=args.url) if args.url else fetch_17500_qlc_text()

    draws = parse_17500_qlc_text(text)
    save_qlc_draws_csv(draws, Path(args.output))
    print(f"已保存 {len(draws)} 条七乐彩开奖数据：{args.output}")


def _normalize_kl8(args) -> None:
    if args.input:
        text = _read_text_file(Path(args.input))
    else:
        text = fetch_17500_kl8_text(url=args.url) if args.url else fetch_17500_kl8_text()

    draws = parse_17500_kl8_text(text)
    save_kl8_draws_csv(draws, Path(args.output))
    print(f"已保存 {len(draws)} 条快乐8开奖数据：{args.output}")


def _normalize_ssq(args) -> None:
    if args.input:
        text = _read_text_file(Path(args.input))
    else:
        text = fetch_17500_ssq_text(url=args.url) if args.url else fetch_17500_ssq_text()

    draws = parse_17500_ssq_text(text)
    save_ssq_draws_csv(draws, Path(args.output))
    print(f"已保存 {len(draws)} 条双色球开奖数据：{args.output}")


def _normalize_dlt(args) -> None:
    if args.input:
        text = _read_text_file(Path(args.input))
    else:
        text = fetch_17500_dlt_text(url=args.url) if args.url else fetch_17500_dlt_text()

    draws = parse_17500_dlt_text(text)
    save_dlt_draws_csv(draws, Path(args.output))
    print(f"已保存 {len(draws)} 条大乐透开奖数据：{args.output}")


def _normalize_pick3(args, fetch_text, parse_text, label: str) -> None:
    if args.input:
        text = _read_text_file(Path(args.input))
    else:
        text = fetch_text(url=args.url) if args.url else fetch_text()

    draws = parse_text(text)
    save_draws_csv(draws, Path(args.output))
    print(f"已保存 {len(draws)} 条{label}开奖数据：{args.output}")


def _update_3d(args) -> None:
    _update_draws(
        args=args,
        fetch_text=fetch_17500_3d_text,
        parse_text=parse_17500_3d_text,
        load_csv=load_draws_csv,
        save_csv=save_draws_csv,
        label="福彩3D",
    )


def _update_pl3(args) -> None:
    _update_draws(
        args=args,
        fetch_text=fetch_17500_pl3_text,
        parse_text=parse_17500_pl3_text,
        load_csv=load_draws_csv,
        save_csv=save_draws_csv,
        label="排列三",
    )


def _update_pl5(args) -> None:
    _update_draws(
        args=args,
        fetch_text=fetch_17500_pl5_text,
        parse_text=parse_17500_pl5_text,
        load_csv=load_pl5_draws_csv,
        save_csv=save_pl5_draws_csv,
        label="排列五",
    )


def _update_qxc(args) -> None:
    _update_draws(
        args=args,
        fetch_text=fetch_17500_qxc_text,
        parse_text=parse_17500_qxc_text,
        load_csv=load_qxc_draws_csv,
        save_csv=save_qxc_draws_csv,
        label="7星彩",
    )


def _update_qlc(args) -> None:
    _update_draws(
        args=args,
        fetch_text=fetch_17500_qlc_text,
        parse_text=parse_17500_qlc_text,
        load_csv=load_qlc_draws_csv,
        save_csv=save_qlc_draws_csv,
        label="七乐彩",
    )


def _update_kl8(args) -> None:
    _update_draws(
        args=args,
        fetch_text=fetch_17500_kl8_text,
        parse_text=parse_17500_kl8_text,
        load_csv=load_kl8_draws_csv,
        save_csv=save_kl8_draws_csv,
        label="快乐8",
    )


def _update_ssq(args) -> None:
    _update_draws(
        args=args,
        fetch_text=fetch_17500_ssq_text,
        parse_text=parse_17500_ssq_text,
        load_csv=load_ssq_draws_csv,
        save_csv=save_ssq_draws_csv,
        label="双色球",
    )


def _update_dlt(args) -> None:
    _update_draws(
        args=args,
        fetch_text=fetch_17500_dlt_text,
        parse_text=parse_17500_dlt_text,
        load_csv=load_dlt_draws_csv,
        save_csv=save_dlt_draws_csv,
        label="大乐透",
    )


def _update_draws(args, fetch_text, parse_text, load_csv, save_csv, label: str) -> None:
    if args.input:
        text = _read_text_file(Path(args.input))
    else:
        text = fetch_text(url=args.url) if args.url else fetch_text()

    source_draws = parse_text(text)
    result = update_draws_csv(
        path=Path(args.csv),
        source_draws=source_draws,
        load_csv=load_csv,
        save_csv=save_csv,
    )
    print(render_incremental_update_result(label, result))


def _backtest_3d(args) -> None:
    _backtest_pick3(args=args, game=Fucai3DGame(), label="福彩3D")


def _backtest_pl3(args) -> None:
    _backtest_pick3(args=args, game=PL3Game(), label="排列三")


def _backtest_pl5(args) -> None:
    draws = load_pl5_draws_csv(Path(args.csv))
    result = run_backtest(
        draws=draws,
        game=PL5Game(),
        strategy=Random5DStrategy(seed=args.seed),
        min_history=args.min_history,
    )
    print(render_backtest_report(result, strategy_name=f"排列五随机基线(seed={args.seed})"))


def _backtest_qxc(args) -> None:
    draws = load_qxc_draws_csv(Path(args.csv))
    result = run_qxc_backtest(
        draws=draws,
        game=QxcGame(),
        strategy=RandomQxcStrategy(seed=args.seed),
        min_history=args.min_history,
    )
    print(render_backtest_report(result, strategy_name=f"7星彩随机基线(seed={args.seed})"))


def _backtest_qlc(args) -> None:
    draws = load_qlc_draws_csv(Path(args.csv))
    result = run_qlc_backtest(
        draws=draws,
        game=QlcGame(),
        strategy=RandomQlcStrategy(seed=args.seed),
        min_history=args.min_history,
    )
    print(render_backtest_report(result, strategy_name=f"七乐彩随机基线(seed={args.seed})"))


def _backtest_kl8(args) -> None:
    draws = load_kl8_draws_csv(Path(args.csv))
    game = Kl8Game(pick_size=args.pick_size)
    result = run_kl8_backtest(
        draws=draws,
        game=game,
        strategy=RandomKl8Strategy(pick_size=args.pick_size, seed=args.seed),
        min_history=args.min_history,
    )
    print(render_backtest_report(result, strategy_name=f"{game.name}随机基线(seed={args.seed})"))


def _backtest_ssq(args) -> None:
    draws = load_ssq_draws_csv(Path(args.csv))
    result = run_ssq_backtest(
        draws=draws,
        game=SsqGame(),
        strategy=RandomSsqStrategy(seed=args.seed),
        min_history=args.min_history,
    )
    print(render_backtest_report(result, strategy_name=f"双色球随机基线(seed={args.seed})"))


def _backtest_dlt(args) -> None:
    draws = load_dlt_draws_csv(Path(args.csv))
    result = run_dlt_backtest(
        draws=draws,
        game=DltGame(),
        strategy=RandomDltStrategy(seed=args.seed),
        min_history=args.min_history,
    )
    print(render_backtest_report(result, strategy_name=f"大乐透随机基线(seed={args.seed})"))


def _backtest_pick3(args, game, label: str) -> None:
    draws = load_draws_csv(Path(args.csv))
    result = run_backtest(
        draws=draws,
        game=game,
        strategy=Random3DStrategy(seed=args.seed),
        min_history=args.min_history,
    )
    print(render_backtest_report(result, strategy_name=f"{label}随机基线(seed={args.seed})"))


def _compare_3d(args) -> None:
    _compare_pick3(args=args, game=Fucai3DGame())


def _compare_pl3(args) -> None:
    _compare_pick3(args=args, game=PL3Game())


def _compare_pl5(args) -> None:
    draws = load_pl5_draws_csv(Path(args.csv))
    strategies = [
        Random5DStrategy(seed=args.seed),
        Hot5DStrategy(window=args.window),
        Cold5DStrategy(window=args.window),
        Omission5DStrategy(window=args.window),
    ]
    results = [
        (
            strategy.name,
            run_backtest(
                draws=draws,
                game=PL5Game(),
                strategy=strategy,
                min_history=args.min_history,
            ),
        )
        for strategy in strategies
    ]
    print(render_compare_report(results))


def _compare_qxc(args) -> None:
    draws = load_qxc_draws_csv(Path(args.csv))
    game = QxcGame()
    strategies = [
        RandomQxcStrategy(seed=args.seed),
        HotQxcStrategy(window=args.window),
        ColdQxcStrategy(window=args.window),
        OmissionQxcStrategy(window=args.window),
    ]
    results = [
        (
            strategy.name,
            run_qxc_backtest(
                draws=draws,
                game=game,
                strategy=strategy,
                min_history=args.min_history,
            ),
        )
        for strategy in strategies
    ]
    print(render_compare_report(results))


def _compare_qlc(args) -> None:
    draws = load_qlc_draws_csv(Path(args.csv))
    game = QlcGame()
    strategies = [
        RandomQlcStrategy(seed=args.seed),
        HotQlcStrategy(window=args.window),
        ColdQlcStrategy(window=args.window),
        OmissionQlcStrategy(window=args.window),
    ]
    results = [
        (
            strategy.name,
            run_qlc_backtest(
                draws=draws,
                game=game,
                strategy=strategy,
                min_history=args.min_history,
            ),
        )
        for strategy in strategies
    ]
    print(render_compare_report(results))


def _compare_kl8(args) -> None:
    draws = load_kl8_draws_csv(Path(args.csv))
    game = Kl8Game(pick_size=args.pick_size)
    strategies = [
        RandomKl8Strategy(pick_size=args.pick_size, seed=args.seed),
        HotKl8Strategy(pick_size=args.pick_size, window=args.window),
        ColdKl8Strategy(pick_size=args.pick_size, window=args.window),
        OmissionKl8Strategy(pick_size=args.pick_size, window=args.window),
    ]
    results = [
        (
            strategy.name,
            run_kl8_backtest(
                draws=draws,
                game=game,
                strategy=strategy,
                min_history=args.min_history,
            ),
        )
        for strategy in strategies
    ]
    print(render_compare_report(results))


def _compare_ssq(args) -> None:
    draws = load_ssq_draws_csv(Path(args.csv))
    game = SsqGame()
    strategies = [
        RandomSsqStrategy(seed=args.seed),
        HotSsqStrategy(window=args.window),
        ColdSsqStrategy(window=args.window),
        OmissionSsqStrategy(window=args.window),
    ]
    results = [
        (
            strategy.name,
            run_ssq_backtest(
                draws=draws,
                game=game,
                strategy=strategy,
                min_history=args.min_history,
            ),
        )
        for strategy in strategies
    ]
    print(render_compare_report(results))


def _compare_dlt(args) -> None:
    draws = load_dlt_draws_csv(Path(args.csv))
    game = DltGame()
    strategies = [
        RandomDltStrategy(seed=args.seed),
        HotDltStrategy(window=args.window),
        ColdDltStrategy(window=args.window),
        OmissionDltStrategy(window=args.window),
    ]
    results = [
        (
            strategy.name,
            run_dlt_backtest(
                draws=draws,
                game=game,
                strategy=strategy,
                min_history=args.min_history,
            ),
        )
        for strategy in strategies
    ]
    print(render_compare_report(results))


def _compare_pick3(args, game) -> None:
    draws = load_draws_csv(Path(args.csv))
    strategies = [
        Random3DStrategy(seed=args.seed),
        Hot3DStrategy(window=args.window),
        Cold3DStrategy(window=args.window),
        Omission3DStrategy(window=args.window),
        SumNearest3DStrategy(window=args.window),
    ]
    results = [
        (
            strategy.name,
            run_backtest(
                draws=draws,
                game=game,
                strategy=strategy,
                min_history=args.min_history,
            ),
        )
        for strategy in strategies
    ]
    print(render_compare_report(results))


def _stability_3d(args) -> None:
    _stability_pick3(args=args, game=Fucai3DGame())


def _stability_pl3(args) -> None:
    _stability_pick3(args=args, game=PL3Game())


def _stability_pl5(args) -> None:
    draws = load_pl5_draws_csv(Path(args.csv))
    seeds = _parse_seed_list(args.seeds)

    seed_results = [
        (
            seed,
            run_backtest(
                draws=draws,
                game=PL5Game(),
                strategy=Random5DStrategy(seed=seed),
                min_history=args.min_history,
            ),
        )
        for seed in seeds
    ]
    seed_sensitivity = summarize_seed_sensitivity(seed_results)

    strategy_sections = []
    for strategy in [
        Hot5DStrategy(window=args.window),
        Cold5DStrategy(window=args.window),
        Omission5DStrategy(window=args.window),
    ]:
        result = run_backtest(
            draws=draws,
            game=PL5Game(),
            strategy=strategy,
            min_history=args.min_history,
        )
        overall = SegmentMetric(
            segment="总体",
            total_bets=result.total_bets,
            total_cost=result.total_cost,
            total_payout=result.total_payout,
            direct_hits=winning_bet_count(result),
        )
        strategy_sections.append((
            strategy.name,
            overall,
            summarize_result_by_issue_year(result),
        ))

    print(render_stability_report(
        seed_sensitivity,
        strategy_sections,
        game_label="排列五",
    ))


def _stability_qxc(args) -> None:
    draws = load_qxc_draws_csv(Path(args.csv))
    seeds = _parse_seed_list(args.seeds)
    game = QxcGame()

    seed_results = [
        (
            seed,
            run_qxc_backtest(
                draws=draws,
                game=game,
                strategy=RandomQxcStrategy(seed=seed),
                min_history=args.min_history,
            ),
        )
        for seed in seeds
    ]
    seed_sensitivity = summarize_seed_sensitivity(seed_results)

    strategy_sections = []
    for strategy in [
        HotQxcStrategy(window=args.window),
        ColdQxcStrategy(window=args.window),
        OmissionQxcStrategy(window=args.window),
    ]:
        result = run_qxc_backtest(
            draws=draws,
            game=game,
            strategy=strategy,
            min_history=args.min_history,
        )
        overall = SegmentMetric(
            segment="总体",
            total_bets=result.total_bets,
            total_cost=result.total_cost,
            total_payout=result.total_payout,
            direct_hits=winning_bet_count(result),
        )
        strategy_sections.append((
            strategy.name,
            overall,
            summarize_result_by_issue_year(result),
        ))

    print(render_stability_report(
        seed_sensitivity,
        strategy_sections,
        game_label="7星彩",
        metric_label="中奖注数",
    ))


def _stability_qlc(args) -> None:
    draws = load_qlc_draws_csv(Path(args.csv))
    seeds = _parse_seed_list(args.seeds)
    game = QlcGame()

    seed_results = [
        (
            seed,
            run_qlc_backtest(
                draws=draws,
                game=game,
                strategy=RandomQlcStrategy(seed=seed),
                min_history=args.min_history,
            ),
        )
        for seed in seeds
    ]
    seed_sensitivity = summarize_seed_sensitivity(seed_results)

    strategy_sections = []
    for strategy in [
        HotQlcStrategy(window=args.window),
        ColdQlcStrategy(window=args.window),
        OmissionQlcStrategy(window=args.window),
    ]:
        result = run_qlc_backtest(
            draws=draws,
            game=game,
            strategy=strategy,
            min_history=args.min_history,
        )
        overall = SegmentMetric(
            segment="总体",
            total_bets=result.total_bets,
            total_cost=result.total_cost,
            total_payout=result.total_payout,
            direct_hits=winning_bet_count(result),
        )
        strategy_sections.append((
            strategy.name,
            overall,
            summarize_result_by_issue_year(result),
        ))

    print(render_stability_report(
        seed_sensitivity,
        strategy_sections,
        game_label="七乐彩",
        metric_label="中奖注数",
    ))


def _stability_kl8(args) -> None:
    draws = load_kl8_draws_csv(Path(args.csv))
    seeds = _parse_seed_list(args.seeds)
    game = Kl8Game(pick_size=args.pick_size)

    seed_results = [
        (
            seed,
            run_kl8_backtest(
                draws=draws,
                game=game,
                strategy=RandomKl8Strategy(pick_size=args.pick_size, seed=seed),
                min_history=args.min_history,
            ),
        )
        for seed in seeds
    ]
    seed_sensitivity = summarize_seed_sensitivity(seed_results)

    strategy_sections = []
    for strategy in [
        HotKl8Strategy(pick_size=args.pick_size, window=args.window),
        ColdKl8Strategy(pick_size=args.pick_size, window=args.window),
        OmissionKl8Strategy(pick_size=args.pick_size, window=args.window),
    ]:
        result = run_kl8_backtest(
            draws=draws,
            game=game,
            strategy=strategy,
            min_history=args.min_history,
        )
        overall = SegmentMetric(
            segment="总体",
            total_bets=result.total_bets,
            total_cost=result.total_cost,
            total_payout=result.total_payout,
            direct_hits=winning_bet_count(result),
        )
        strategy_sections.append((
            strategy.name,
            overall,
            summarize_result_by_issue_year(result),
        ))

    print(render_stability_report(
        seed_sensitivity,
        strategy_sections,
        game_label=game.name,
        metric_label="中奖注数",
    ))


def _stability_ssq(args) -> None:
    draws = load_ssq_draws_csv(Path(args.csv))
    seeds = _parse_seed_list(args.seeds)
    game = SsqGame()

    seed_results = [
        (
            seed,
            run_ssq_backtest(
                draws=draws,
                game=game,
                strategy=RandomSsqStrategy(seed=seed),
                min_history=args.min_history,
            ),
        )
        for seed in seeds
    ]
    seed_sensitivity = summarize_seed_sensitivity(seed_results)

    strategy_sections = []
    for strategy in [
        HotSsqStrategy(window=args.window),
        ColdSsqStrategy(window=args.window),
        OmissionSsqStrategy(window=args.window),
    ]:
        result = run_ssq_backtest(
            draws=draws,
            game=game,
            strategy=strategy,
            min_history=args.min_history,
        )
        overall = SegmentMetric(
            segment="总体",
            total_bets=result.total_bets,
            total_cost=result.total_cost,
            total_payout=result.total_payout,
            direct_hits=winning_bet_count(result),
        )
        strategy_sections.append((
            strategy.name,
            overall,
            summarize_result_by_issue_year(result),
        ))

    print(render_stability_report(
        seed_sensitivity,
        strategy_sections,
        game_label="双色球",
        metric_label="中奖注数",
    ))


def _stability_dlt(args) -> None:
    draws = load_dlt_draws_csv(Path(args.csv))
    seeds = _parse_seed_list(args.seeds)
    game = DltGame()

    seed_results = [
        (
            seed,
            run_dlt_backtest(
                draws=draws,
                game=game,
                strategy=RandomDltStrategy(seed=seed),
                min_history=args.min_history,
            ),
        )
        for seed in seeds
    ]
    seed_sensitivity = summarize_seed_sensitivity(seed_results)

    strategy_sections = []
    for strategy in [
        HotDltStrategy(window=args.window),
        ColdDltStrategy(window=args.window),
        OmissionDltStrategy(window=args.window),
    ]:
        result = run_dlt_backtest(
            draws=draws,
            game=game,
            strategy=strategy,
            min_history=args.min_history,
        )
        overall = SegmentMetric(
            segment="总体",
            total_bets=result.total_bets,
            total_cost=result.total_cost,
            total_payout=result.total_payout,
            direct_hits=winning_bet_count(result),
        )
        strategy_sections.append((
            strategy.name,
            overall,
            summarize_result_by_issue_year(result),
        ))

    print(render_stability_report(
        seed_sensitivity,
        strategy_sections,
        game_label="大乐透",
        metric_label="中奖注数",
    ))


def _recommend_3d(args) -> None:
    _recommend_pick3(args=args, game=Fucai3DGame())


def _recommend_pl3(args) -> None:
    _recommend_pick3(args=args, game=PL3Game())


def _recommend_pl5(args) -> None:
    draws = load_pl5_draws_csv(Path(args.csv))
    random_strategy = Random5DStrategy(seed=args.seed)
    _print_recommendation_report(
        draws=draws,
        game=PL5Game(),
        strategies=[
            random_strategy,
            Hot5DStrategy(window=args.window),
            Cold5DStrategy(window=args.window),
            Omission5DStrategy(window=args.window),
        ],
        filler_strategy=random_strategy,
        count=args.count,
    )


def _recommend_qxc(args) -> None:
    draws = load_qxc_draws_csv(Path(args.csv))
    random_strategy = RandomQxcStrategy(seed=args.seed)
    _print_recommendation_report(
        draws=draws,
        game=QxcGame(),
        strategies=[
            random_strategy,
            HotQxcStrategy(window=args.window),
            ColdQxcStrategy(window=args.window),
            OmissionQxcStrategy(window=args.window),
        ],
        filler_strategy=random_strategy,
        count=args.count,
    )


def _recommend_qlc(args) -> None:
    draws = load_qlc_draws_csv(Path(args.csv))
    random_strategy = RandomQlcStrategy(seed=args.seed)
    _print_recommendation_report(
        draws=draws,
        game=QlcGame(),
        strategies=[
            random_strategy,
            HotQlcStrategy(window=args.window),
            ColdQlcStrategy(window=args.window),
            OmissionQlcStrategy(window=args.window),
        ],
        filler_strategy=random_strategy,
        count=args.count,
    )


def _recommend_kl8(args) -> None:
    draws = load_kl8_draws_csv(Path(args.csv))
    game = Kl8Game(pick_size=args.pick_size)
    random_strategy = RandomKl8Strategy(pick_size=args.pick_size, seed=args.seed)
    _print_recommendation_report(
        draws=draws,
        game=game,
        strategies=[
            random_strategy,
            HotKl8Strategy(pick_size=args.pick_size, window=args.window),
            ColdKl8Strategy(pick_size=args.pick_size, window=args.window),
            OmissionKl8Strategy(pick_size=args.pick_size, window=args.window),
        ],
        filler_strategy=random_strategy,
        count=args.count,
    )


def _recommend_ssq(args) -> None:
    draws = load_ssq_draws_csv(Path(args.csv))
    random_strategy = RandomSsqStrategy(seed=args.seed)
    _print_recommendation_report(
        draws=draws,
        game=SsqGame(),
        strategies=[
            random_strategy,
            HotSsqStrategy(window=args.window),
            ColdSsqStrategy(window=args.window),
            OmissionSsqStrategy(window=args.window),
        ],
        filler_strategy=random_strategy,
        count=args.count,
    )


def _recommend_dlt(args) -> None:
    draws = load_dlt_draws_csv(Path(args.csv))
    random_strategy = RandomDltStrategy(seed=args.seed)
    _print_recommendation_report(
        draws=draws,
        game=DltGame(),
        strategies=[
            random_strategy,
            HotDltStrategy(window=args.window),
            ColdDltStrategy(window=args.window),
            OmissionDltStrategy(window=args.window),
        ],
        filler_strategy=random_strategy,
        count=args.count,
    )


def _train_ml_ssq(args) -> None:
    draws = available_recommendation_draws(load_ssq_draws_csv(Path(args.csv)))
    model = train_ssq_ml_model(
        draws,
        min_history=args.min_history,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    save_ssq_ml_model(model, Path(args.model))
    print("双色球机器学习模型已保存")
    print(f"model: {args.model}")
    print(f"training_draws: {model.training_draw_count}")
    print(f"training_targets: {model.training_target_count}")
    print(f"features: {len(model.feature_names)}")


def _backtest_ml_ssq(args) -> None:
    draws = available_recommendation_draws(load_ssq_draws_csv(Path(args.csv)))
    result = run_ssq_ml_backtest(
        draws,
        min_train=args.min_train,
        limit=args.limit,
        retrain_every=args.retrain_every,
        min_history=args.min_history,
        epochs=args.epochs,
    )
    print(render_backtest_report(result, strategy_name="双色球机器学习"))


def _recommend_ml_ssq(args) -> None:
    draws = available_recommendation_draws(load_ssq_draws_csv(Path(args.csv)))
    model_path = Path(args.model)
    if model_path.exists():
        model = load_ssq_ml_model(model_path)
    else:
        model = train_ssq_ml_model(draws, min_history=args.min_history, epochs=args.epochs)
        save_ssq_ml_model(model, model_path)
    candidates = recommend_ssq_ml(draws, model, count=args.count)
    ordered_draws = tuple(sorted(draws, key=lambda draw: int(draw.issue)))
    latest_issue = ordered_draws[-1].issue if ordered_draws else ""
    print(render_recommendation_report(
        game_name="双色球机器学习",
        candidates=candidates,
        history_count=len(ordered_draws),
        latest_issue=latest_issue,
    ))


def _train_ml_3d(args) -> None:
    _train_ml_generic(args, "3d")


def _train_ml_pl3(args) -> None:
    _train_ml_generic(args, "pl3")


def _train_ml_pl5(args) -> None:
    _train_ml_generic(args, "pl5")


def _train_ml_qxc(args) -> None:
    _train_ml_generic(args, "qxc")


def _train_ml_qlc(args) -> None:
    _train_ml_generic(args, "qlc")


def _train_ml_kl8(args) -> None:
    _train_ml_generic(args, "kl8")


def _train_ml_dlt(args) -> None:
    _train_ml_generic(args, "dlt")


def _backtest_ml_3d(args) -> None:
    _backtest_ml_generic(args, "3d")


def _backtest_ml_pl3(args) -> None:
    _backtest_ml_generic(args, "pl3")


def _backtest_ml_pl5(args) -> None:
    _backtest_ml_generic(args, "pl5")


def _backtest_ml_qxc(args) -> None:
    _backtest_ml_generic(args, "qxc")


def _backtest_ml_qlc(args) -> None:
    _backtest_ml_generic(args, "qlc")


def _backtest_ml_kl8(args) -> None:
    _backtest_ml_generic(args, "kl8")


def _backtest_ml_dlt(args) -> None:
    _backtest_ml_generic(args, "dlt")


def _recommend_ml_3d(args) -> None:
    _recommend_ml_generic(args, "3d")


def _recommend_ml_pl3(args) -> None:
    _recommend_ml_generic(args, "pl3")


def _recommend_ml_pl5(args) -> None:
    _recommend_ml_generic(args, "pl5")


def _recommend_ml_qxc(args) -> None:
    _recommend_ml_generic(args, "qxc")


def _recommend_ml_qlc(args) -> None:
    _recommend_ml_generic(args, "qlc")


def _recommend_ml_kl8(args) -> None:
    _recommend_ml_generic(args, "kl8")


def _recommend_ml_dlt(args) -> None:
    _recommend_ml_generic(args, "dlt")


def _record_recommend_ml_3d(args) -> None:
    _record_recommend_ml_generic(args, "3d")


def _record_recommend_ml_pl3(args) -> None:
    _record_recommend_ml_generic(args, "pl3")


def _record_recommend_ml_pl5(args) -> None:
    _record_recommend_ml_generic(args, "pl5")


def _record_recommend_ml_qxc(args) -> None:
    _record_recommend_ml_generic(args, "qxc")


def _record_recommend_ml_qlc(args) -> None:
    _record_recommend_ml_generic(args, "qlc")


def _record_recommend_ml_kl8(args) -> None:
    _record_recommend_ml_generic(args, "kl8")


def _record_recommend_ml_dlt(args) -> None:
    _record_recommend_ml_generic(args, "dlt")


def _train_ml_generic(args, game_code: str) -> None:
    draws, game, adapter = _load_generic_ml_context(args, game_code)
    model = train_generic_ml_model(
        draws,
        adapter,
        min_history=args.min_history,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    save_generic_ml_model(model, Path(args.model))
    print(f"{game.name}机器学习模型已保存")
    print(f"model: {args.model}")
    print(f"training_draws: {model.training_draw_count}")
    print(f"training_targets: {model.training_target_count}")
    print(f"features: {len(model.feature_names)}")


def _backtest_ml_generic(args, game_code: str) -> None:
    draws, game, adapter = _load_generic_ml_context(args, game_code)
    result = run_generic_ml_backtest(
        draws,
        adapter,
        game,
        min_train=args.min_train,
        limit=args.limit,
        retrain_every=args.retrain_every,
        min_history=args.min_history,
        epochs=args.epochs,
    )
    print(render_backtest_report(result, strategy_name=f"{game.name}机器学习"))


def _recommend_ml_generic(args, game_code: str) -> None:
    draws, game, adapter = _load_generic_ml_context(args, game_code)
    model_path = Path(args.model)
    if model_path.exists():
        model = load_generic_ml_model(model_path)
    else:
        model = train_generic_ml_model(draws, adapter, min_history=args.min_history, epochs=args.epochs)
        save_generic_ml_model(model, model_path)
    candidates = recommend_generic_ml(draws, model, adapter, count=args.count)
    ordered_draws = tuple(sorted(draws, key=lambda draw: int(draw.issue)))
    latest_issue = ordered_draws[-1].issue if ordered_draws else ""
    print(render_recommendation_report(
        game_name=f"{game.name}机器学习",
        candidates=candidates,
        history_count=len(ordered_draws),
        latest_issue=latest_issue,
    ))


def _record_recommend_ml_generic(args, game_code: str) -> None:
    draws, game, adapter = _load_generic_ml_context(args, game_code)
    target_issue = _resolve_recommendation_target_issue(
        game_code=game_code,
        draws=draws,
        target_issue=args.target_issue,
        history_until_issue=args.history_until,
    )
    window = select_recommendation_window(
        draws=draws,
        target_issue=target_issue,
        history_until_issue=args.history_until,
    )
    model_path = Path(args.model)
    if model_path.exists() and not args.target_issue and not args.history_until:
        model = load_generic_ml_model(model_path)
    else:
        model = train_generic_ml_model(window.history, adapter, min_history=args.min_history, epochs=args.epochs)
        if not args.target_issue and not args.history_until:
            save_generic_ml_model(model, model_path)

    candidates = recommend_generic_ml(window.history, model, adapter, count=args.count)
    records = create_recommendation_records(
        game_code=game_code,
        game_name=game.name,
        candidates=candidates,
        target_issue=window.target_issue,
        history_until_issue=window.history_until_issue,
        ticket_cost=game.ticket_cost,
    )

    if args.output:
        save_recommendation_records(records, Path(args.output))
        print(f"{game.name} machine-learning recommendation records saved: {args.output}")
        print(f"history_until_issue: {window.history_until_issue}")
        print(f"target_issue: {window.target_issue}")
        print(f"records: {len(records)}")
        return

    path, added, total = RecommendationStore(Path(args.store)).append_records(records)
    print(f"{game.name} machine-learning recommendation records saved: {path}")
    print(f"history_until_issue: {window.history_until_issue}")
    print(f"target_issue: {window.target_issue}")
    print(f"added_records: {added}")
    print(f"total_records: {total}")


def _load_generic_ml_context(args, game_code: str):
    if game_code == "3d":
        return (
            available_recommendation_draws(load_draws_csv(Path(args.csv))),
            Fucai3DGame(),
            ml_adapter_for_game("3d"),
        )
    if game_code == "pl3":
        return (
            available_recommendation_draws(load_draws_csv(Path(args.csv))),
            PL3Game(),
            ml_adapter_for_game("pl3"),
        )
    if game_code == "pl5":
        return (
            available_recommendation_draws(load_pl5_draws_csv(Path(args.csv))),
            PL5Game(),
            ml_adapter_for_game("pl5"),
        )
    if game_code == "qxc":
        return (
            available_recommendation_draws(load_qxc_draws_csv(Path(args.csv))),
            QxcGame(),
            ml_adapter_for_game("qxc"),
        )
    if game_code == "qlc":
        return (
            available_recommendation_draws(load_qlc_draws_csv(Path(args.csv))),
            QlcGame(),
            ml_adapter_for_game("qlc"),
        )
    if game_code == "kl8":
        pick_size = getattr(args, "pick_size", 10)
        return (
            available_recommendation_draws(load_kl8_draws_csv(Path(args.csv))),
            Kl8Game(pick_size=pick_size),
            ml_adapter_for_game("kl8", pick_size=pick_size),
        )
    if game_code == "dlt":
        return (
            available_recommendation_draws(load_dlt_draws_csv(Path(args.csv))),
            DltGame(),
            ml_adapter_for_game("dlt"),
        )
    raise ValueError(f"unsupported ML game: {game_code}")


def _recommend_pick3(args, game) -> None:
    draws = load_draws_csv(Path(args.csv))
    random_strategy = Random3DStrategy(seed=args.seed)
    _print_recommendation_report(
        draws=draws,
        game=game,
        strategies=[
            random_strategy,
            Hot3DStrategy(window=args.window),
            Cold3DStrategy(window=args.window),
            Omission3DStrategy(window=args.window),
            SumNearest3DStrategy(window=args.window),
        ],
        filler_strategy=random_strategy,
        count=args.count,
    )


def _print_recommendation_report(draws, game, strategies, filler_strategy, count: int) -> None:
    ordered_draws = tuple(sorted(available_recommendation_draws(draws), key=lambda draw: int(draw.issue)))
    candidates = generate_candidates(
        history=ordered_draws,
        game=game,
        strategies=strategies,
        filler_strategy=filler_strategy,
        count=count,
    )
    latest_issue = ordered_draws[-1].issue if ordered_draws else ""
    print(render_recommendation_report(
        game_name=game.name,
        candidates=candidates,
        history_count=len(ordered_draws),
        latest_issue=latest_issue,
    ))


def _record_recommend_3d(args) -> None:
    _record_recommend_pick3(args=args, game_code="3d", game=Fucai3DGame())


def _record_recommend_pl3(args) -> None:
    _record_recommend_pick3(args=args, game_code="pl3", game=PL3Game())


def _record_recommend_pl5(args) -> None:
    draws = load_pl5_draws_csv(Path(args.csv))
    random_strategy = Random5DStrategy(seed=args.seed)
    _save_recommendation_records(
        args=args,
        game_code="pl5",
        draws=draws,
        game=PL5Game(),
        strategies=[
            random_strategy,
            Hot5DStrategy(window=args.window),
            Cold5DStrategy(window=args.window),
            Omission5DStrategy(window=args.window),
        ],
        filler_strategy=random_strategy,
    )


def _record_recommend_qxc(args) -> None:
    draws = load_qxc_draws_csv(Path(args.csv))
    random_strategy = RandomQxcStrategy(seed=args.seed)
    _save_recommendation_records(
        args=args,
        game_code="qxc",
        draws=draws,
        game=QxcGame(),
        strategies=[
            random_strategy,
            HotQxcStrategy(window=args.window),
            ColdQxcStrategy(window=args.window),
            OmissionQxcStrategy(window=args.window),
        ],
        filler_strategy=random_strategy,
    )


def _record_recommend_qlc(args) -> None:
    draws = load_qlc_draws_csv(Path(args.csv))
    random_strategy = RandomQlcStrategy(seed=args.seed)
    _save_recommendation_records(
        args=args,
        game_code="qlc",
        draws=draws,
        game=QlcGame(),
        strategies=[
            random_strategy,
            HotQlcStrategy(window=args.window),
            ColdQlcStrategy(window=args.window),
            OmissionQlcStrategy(window=args.window),
        ],
        filler_strategy=random_strategy,
    )


def _record_recommend_kl8(args) -> None:
    draws = load_kl8_draws_csv(Path(args.csv))
    game = Kl8Game(pick_size=args.pick_size)
    random_strategy = RandomKl8Strategy(pick_size=args.pick_size, seed=args.seed)
    _save_recommendation_records(
        args=args,
        game_code="kl8",
        draws=draws,
        game=game,
        strategies=[
            random_strategy,
            HotKl8Strategy(pick_size=args.pick_size, window=args.window),
            ColdKl8Strategy(pick_size=args.pick_size, window=args.window),
            OmissionKl8Strategy(pick_size=args.pick_size, window=args.window),
        ],
        filler_strategy=random_strategy,
    )


def _record_recommend_ssq(args) -> None:
    draws = load_ssq_draws_csv(Path(args.csv))
    random_strategy = RandomSsqStrategy(seed=args.seed)
    _save_recommendation_records(
        args=args,
        game_code="ssq",
        draws=draws,
        game=SsqGame(),
        strategies=[
            random_strategy,
            HotSsqStrategy(window=args.window),
            ColdSsqStrategy(window=args.window),
            OmissionSsqStrategy(window=args.window),
        ],
        filler_strategy=random_strategy,
    )


def _record_recommend_ml_ssq(args) -> None:
    draws = available_recommendation_draws(load_ssq_draws_csv(Path(args.csv)))
    target_issue = _resolve_recommendation_target_issue(
        game_code="ssq",
        draws=draws,
        target_issue=args.target_issue,
        history_until_issue=args.history_until,
    )
    window = select_recommendation_window(
        draws=draws,
        target_issue=target_issue,
        history_until_issue=args.history_until,
    )
    model_path = Path(args.model)
    if model_path.exists() and not args.target_issue and not args.history_until:
        model = load_ssq_ml_model(model_path)
    else:
        model = train_ssq_ml_model(window.history, min_history=args.min_history, epochs=args.epochs)
        if not args.target_issue and not args.history_until:
            save_ssq_ml_model(model, model_path)

    game = SsqGame()
    candidates = recommend_ssq_ml(window.history, model, count=args.count)
    records = create_recommendation_records(
        game_code="ssq",
        game_name=game.name,
        candidates=candidates,
        target_issue=window.target_issue,
        history_until_issue=window.history_until_issue,
        ticket_cost=game.ticket_cost,
    )

    if args.output:
        save_recommendation_records(records, Path(args.output))
        print(f"{game.name} machine-learning recommendation records saved: {args.output}")
        print(f"history_until_issue: {window.history_until_issue}")
        print(f"target_issue: {window.target_issue}")
        print(f"records: {len(records)}")
        return

    path, added, total = RecommendationStore(Path(args.store)).append_records(records)
    print(f"{game.name} machine-learning recommendation records saved: {path}")
    print(f"history_until_issue: {window.history_until_issue}")
    print(f"target_issue: {window.target_issue}")
    print(f"added_records: {added}")
    print(f"total_records: {total}")


def _record_recommend_dlt(args) -> None:
    draws = load_dlt_draws_csv(Path(args.csv))
    random_strategy = RandomDltStrategy(seed=args.seed)
    _save_recommendation_records(
        args=args,
        game_code="dlt",
        draws=draws,
        game=DltGame(),
        strategies=[
            random_strategy,
            HotDltStrategy(window=args.window),
            ColdDltStrategy(window=args.window),
            OmissionDltStrategy(window=args.window),
        ],
        filler_strategy=random_strategy,
    )


def _record_recommend_pick3(args, game_code: str, game) -> None:
    draws = load_draws_csv(Path(args.csv))
    random_strategy = Random3DStrategy(seed=args.seed)
    _save_recommendation_records(
        args=args,
        game_code=game_code,
        draws=draws,
        game=game,
        strategies=[
            random_strategy,
            Hot3DStrategy(window=args.window),
            Cold3DStrategy(window=args.window),
            Omission3DStrategy(window=args.window),
            SumNearest3DStrategy(window=args.window),
        ],
        filler_strategy=random_strategy,
    )


def _resolve_recommendation_target_issue(
    game_code: str,
    draws,
    target_issue: str = "",
    history_until_issue: str = "",
) -> str:
    if target_issue:
        return target_issue

    ordered = tuple(sorted(draws, key=lambda draw: int(draw.issue)))
    if not ordered:
        raise ValueError("No draws available")

    if history_until_issue:
        matches = [draw for draw in ordered if draw.issue == history_until_issue]
        if not matches:
            raise ValueError(f"History issue not found: {history_until_issue}")
        anchor = matches[-1]
    else:
        anchor = ordered[-1]

    return next_issue_from_latest_draw(game_code, anchor.issue, anchor.draw_date).issue


def _save_recommendation_records(args, game_code: str, draws, game, strategies, filler_strategy) -> None:
    draws = available_recommendation_draws(draws)
    target_issue = _resolve_recommendation_target_issue(
        game_code=game_code,
        draws=draws,
        target_issue=args.target_issue,
        history_until_issue=args.history_until,
    )
    window = select_recommendation_window(
        draws=draws,
        target_issue=target_issue,
        history_until_issue=args.history_until,
    )
    candidates = generate_candidates(
        history=window.history,
        game=game,
        strategies=strategies,
        filler_strategy=filler_strategy,
        count=args.count,
    )
    records = create_recommendation_records(
        game_code=game_code,
        game_name=game.name,
        candidates=candidates,
        target_issue=window.target_issue,
        history_until_issue=window.history_until_issue,
        ticket_cost=game.ticket_cost,
    )
    if args.output:
        save_recommendation_records(records, Path(args.output))
        print(f"{game.name} recommendation records saved: {args.output}")
        print(f"history_until_issue: {window.history_until_issue}")
        print(f"target_issue: {window.target_issue}")
        print(f"records: {len(records)}")
        return

    path, added, total = RecommendationStore(Path(args.store)).append_records(records)
    print(f"{game.name} recommendation records saved: {path}")
    print(f"history_until_issue: {window.history_until_issue}")
    print(f"target_issue: {window.target_issue}")
    print(f"added_records: {added}")
    print(f"total_records: {total}")
    return


def _verify_recommend_3d(args) -> None:
    _verify_recommend(args=args, game_code="3d", draws=load_draws_csv(Path(args.csv)), game=Fucai3DGame())


def _verify_recommend_pl3(args) -> None:
    _verify_recommend(args=args, game_code="pl3", draws=load_draws_csv(Path(args.csv)), game=PL3Game())


def _verify_recommend_pl5(args) -> None:
    _verify_recommend(args=args, game_code="pl5", draws=load_pl5_draws_csv(Path(args.csv)), game=PL5Game())


def _verify_recommend_qxc(args) -> None:
    _verify_recommend(args=args, game_code="qxc", draws=load_qxc_draws_csv(Path(args.csv)), game=QxcGame())


def _verify_recommend_qlc(args) -> None:
    _verify_recommend(args=args, game_code="qlc", draws=load_qlc_draws_csv(Path(args.csv)), game=QlcGame())


def _verify_recommend_kl8(args) -> None:
    _verify_recommend(
        args=args,
        game_code="kl8",
        draws=load_kl8_draws_csv(Path(args.csv)),
        game=Kl8Game(pick_size=args.pick_size),
    )


def _verify_recommend_ssq(args) -> None:
    _verify_recommend(args=args, game_code="ssq", draws=load_ssq_draws_csv(Path(args.csv)), game=SsqGame())


def _verify_recommend_dlt(args) -> None:
    _verify_recommend(args=args, game_code="dlt", draws=load_dlt_draws_csv(Path(args.csv)), game=DltGame())


def _verify_recommend(args, game_code: str, draws, game) -> None:
    records = load_recommendation_records(Path(args.records))
    result = evaluate_recommendation_records(
        records=records,
        draws=draws,
        game=game,
        pick_parser=lambda value: parse_pick_text(game_code, value),
    )
    if args.output:
        save_recommendation_records(result.records, Path(args.output))
    print(render_recommendation_verification_report(result))


def _summarize_recommendations(args) -> None:
    records = load_records_from_paths([Path(path) for path in args.records])
    result = summarize_recommendation_records(records)
    report = render_recommendation_summary_report(result)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report, encoding="utf-8")
    print(report)


def _dashboard(args) -> None:
    if args.server == "fastapi":
        serve_fastapi_dashboard(
            reports_dir=Path(args.reports),
            host=args.host,
            port=args.port,
            open_browser=args.open_browser,
        )
        return
    if args.server == "auto":
        try:
            serve_fastapi_dashboard(
                reports_dir=Path(args.reports),
                host=args.host,
                port=args.port,
                open_browser=args.open_browser,
            )
            return
        except RuntimeError as exc:
            print(f"{exc}，已切换到内置本地服务。")
    serve_dashboard(
        reports_dir=Path(args.reports),
        host=args.host,
        port=args.port,
        open_browser=args.open_browser,
    )


def _stability_pick3(args, game) -> None:
    draws = load_draws_csv(Path(args.csv))
    seeds = _parse_seed_list(args.seeds)

    seed_results = [
        (
            seed,
            run_backtest(
                draws=draws,
                game=game,
                strategy=Random3DStrategy(seed=seed),
                min_history=args.min_history,
            ),
        )
        for seed in seeds
    ]
    seed_sensitivity = summarize_seed_sensitivity(seed_results)

    strategy_sections = []
    for strategy in [
        Hot3DStrategy(window=args.window),
        Cold3DStrategy(window=args.window),
        Omission3DStrategy(window=args.window),
        SumNearest3DStrategy(window=args.window),
    ]:
        result = run_backtest(
            draws=draws,
            game=game,
            strategy=strategy,
            min_history=args.min_history,
        )
        overall = SegmentMetric(
            segment="总体",
            total_bets=result.total_bets,
            total_cost=result.total_cost,
            total_payout=result.total_payout,
            direct_hits=winning_bet_count(result),
        )
        strategy_sections.append((
            strategy.name,
            overall,
            summarize_result_by_issue_year(result),
        ))

    print(render_stability_report(
        seed_sensitivity,
        strategy_sections,
        game_label=_game_label(game),
    ))


def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _parse_seed_list(value: str) -> Sequence[int]:
    seeds = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    if not seeds:
        raise ValueError("至少需要一个随机种子")
    return seeds


def _game_label(game) -> str:
    return game.name.replace("直选", "").strip()


if __name__ == "__main__":
    raise SystemExit(main())
