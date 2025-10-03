import argparse
import json
import time
from datetime import timedelta
from typing import Any, Dict

from llama_index.core.llms import LLM

from mage.agent import TopAgent
from mage.benchmark_read_helper import (
    TypeBenchmark,
    TypeBenchmarkFile,
    get_benchmark_contents,
)
from mage.gen_config import get_llm, set_exp_setting
from mage.log_utils import get_logger
from mage.token_counter import TokenCount

logger = get_logger(__name__)

DEFAULTS = {
    "provider": "openai",
    "model": "gpt-4o-2024-08-06",
    "filter_instance": "^(Prob006_stream_xbar)$",
    "type_benchmark": "pulp_verilog_eval",
    "path_benchmark": "./pulp-verilog-eval",
    "run_identifier": "exp00",
    "n": 1,
    "temperature": 0.85,
    "top_p": 0.95,
    "max_token": 8192,
    "use_golden_tb_in_mage": True,
    "key_cfg_path": "./key.cfg",
    "simulator": "questa",
    "input_spec": None,
    "input_rtl": None,
    "rtl_top": None,
    "input_tb": None,
}


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Unifies the 'benchmark suite' and 'spare design' frontends into one CLI.

    Defaults mirror the previously hardcoded args_dict_benchmark_suite.
    Spare design mode triggers if --input-spec, --input-tb, and --rtl-top are provided.
    --input-rtl is OPTIONAL in spare design mode.
    """
    p = argparse.ArgumentParser(
        description="Unified runner for benchmark suite or spare design."
    )
    # Common / provider & model
    p.add_argument("--provider", default=DEFAULTS["provider"], help="LLM provider.")
    p.add_argument("--model", default=DEFAULTS["model"], help="LLM model name.")
    p.add_argument(
        "--key-cfg-path",
        default=DEFAULTS["key_cfg_path"],
        help="Path to API key/config file.",
    )
    p.add_argument(
        "--max-token", type=int, default=DEFAULTS["max_token"], help="Max tokens."
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=DEFAULTS["temperature"],
        help="Sampling temperature.",
    )
    p.add_argument("--top-p", type=float, default=DEFAULTS["top_p"], help="Top-p.")
    p.add_argument(
        "--run-identifier",
        default=DEFAULTS["run_identifier"],
        help="Run identifier prefix.",
    )
    p.add_argument("-n", type=int, default=DEFAULTS["n"], help="Number of rounds.")
    p.add_argument(
        "--use-golden-tb-in-mage",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["use_golden_tb_in_mage"],
        help="Whether to pass golden TB/RTL paths into mage (bool).",
    )
    p.add_argument(
        "--simulator",
        default=DEFAULTS["simulator"],
        help="Simulator backend (e.g., questa).",
    )

    # Benchmark suite options
    p.add_argument(
        "--type-benchmark",
        default=DEFAULTS["type_benchmark"],
        help="Benchmark type key (e.g., pulp_verilog_eval).",
    )
    p.add_argument(
        "--path-benchmark",
        default=DEFAULTS["path_benchmark"],
        help="Path to benchmark root.",
    )
    p.add_argument(
        "--filter-instance",
        default=DEFAULTS["filter_instance"],
        help="Regex to filter which instances to run (for benchmark mode).",
    )

    # Spare design options
    p.add_argument(
        "--input-spec",
        default=DEFAULTS["input_spec"],
        help="Path to a single design spec file.",
    )
    p.add_argument(
        "--input-rtl",
        default=DEFAULTS["input_rtl"],
        help="Path to golden RTL (blackbox) file/folder. Optional.",
    )
    p.add_argument(
        "--rtl-top",
        default=DEFAULTS["rtl_top"],
        help="Top module name for the spare design.",
    )
    p.add_argument(
        "--input-tb",
        default=DEFAULTS["input_tb"],
        help="Path to golden testbench for the spare design.",
    )

    return p


def run_round(
    args: argparse.Namespace,
    llm: LLM,
    spec_dict: Dict[str, str],
    golden_tb_path_dict: Dict[str, str],
    golden_rtl_path_dict: Dict[str, str],
):
    total_start_time = time.monotonic()

    logger.info(spec_dict)
    logger.info(golden_tb_path_dict)
    logger.info(golden_rtl_path_dict)

    agent = TopAgent(llm)
    agent.set_output_path(f"./output_{args.run_identifier}")
    agent.set_log_path(f"./log_{args.run_identifier}")
    agent.set_redirect_log(False)
    # agent.set_ablation(True)
    record_file = f"./output_{args.run_identifier}/record.json"
    record_json: Dict[str, Dict[str, Any]] = {"record_per_run": {}, "total_record": {}}

    ret: dict[str, tuple[bool, str]] = {}
    pass_cnt = 0
    token_sum = TokenCount(in_token_cnt=0, out_token_cnt=0)
    token_limit_cnt = 0
    for i, (task_id, spec) in enumerate(spec_dict.items()):
        start_time = time.monotonic()
        print(f"({i+1:03d}/{len(spec_dict):03d}) Current task: {task_id}")
        ret[task_id] = agent.run(
            simulator=args.simulator,
            log_prefix=(
                args.type_benchmark.name
                if args.type_benchmark.name is not None
                else args.rtl_top
            ),  # noqa: F821
            task_id=task_id,
            spec=spec,
            golden_tb_path=(
                golden_tb_path_dict[task_id] if args.use_golden_tb_in_mage else None
            ),
            golden_rtl_blackbox_path=(
                golden_rtl_path_dict[task_id] if args.use_golden_tb_in_mage else None
            ),
        )
        is_pass = ret[task_id][0]
        run_time = timedelta(seconds=time.monotonic() - start_time)
        print(f"{task_id} took {run_time} to execute")
        print(f"({i+1:03d}/{len(spec_dict):03d}) {task_id}: is_pass = {is_pass}")
        run_token_cnt = agent.token_counter.get_sum_count()
        print(
            f"Current problem token count: Input {run_token_cnt.in_token_cnt}, Output {run_token_cnt.out_token_cnt}"
        )
        if agent.token_counter.token_cost:
            run_cost = (
                run_token_cnt.in_token_cnt
                * agent.token_counter.token_cost.in_token_cost_per_token
                + run_token_cnt.out_token_cnt
                * agent.token_counter.token_cost.out_token_cost_per_token
            )
        run_token_limit_cnt = agent.token_counter.get_total_token()
        print(f"Current problem token limit consumption: {run_token_limit_cnt}")
        token_limit_cnt += run_token_limit_cnt
        print(f"{'Current problem token cost':<25}: ${run_cost:.2f} USD")
        token_sum += run_token_cnt
        pass_cnt += is_pass
        record_json["record_per_run"][task_id] = {
            "is_pass": is_pass,
            "run_token_limit_cnt": f"{run_token_limit_cnt:.2f}",
            "run_token_cost": f"{run_cost:.2f}",
            "run_time": str(run_time),
        }
    print(f"Pass rate: {pass_cnt}/{len(spec_dict)}")
    print(
        f"Total token count: Input {token_sum.in_token_cnt}, Output {token_sum.out_token_cnt}"
    )
    print(f"Total token limit consumption: {token_limit_cnt}")
    if agent.token_counter.token_cost:
        total_cost = (
            token_sum.in_token_cnt
            * agent.token_counter.token_cost.in_token_cost_per_token
            + token_sum.out_token_cnt
            * agent.token_counter.token_cost.out_token_cost_per_token
        )
        print(f"{'Total cost':<25}: ${total_cost:.2f} USD")
        print(f"{'Avg cost':<25}: ${total_cost / len(spec_dict):.2f} USD")

    total_run_time = timedelta(seconds=time.monotonic() - total_start_time)
    print(f"Totally took {total_run_time} to execute")
    record_json["total_record"] = {
        "pass_cnt": pass_cnt,
        "total_cnt": len(spec_dict),
        "token_limit_cnt": token_limit_cnt,
        "total_cost": f"{total_cost:.2f}",
        "avg_cost": f"{total_cost / len(spec_dict):.2f}",
        "total_run_time": str(total_run_time),
    }
    json.dump(record_json, open(record_file, "w"), indent=4)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Determine mode
    spare_design_mode = all([args.input_spec, args.input_tb, args.rtl_top])

    if spare_design_mode:
        with open(args.input_spec, "r") as f:
            input_spec_str = f.read()

        spec_dict = {args.rtl_top: input_spec_str}
        golden_tb_path_dict = {args.rtl_top: args.input_tb}
        golden_rtl_path_dict = {args.rtl_top: args.input_rtl}

        # Make names visible to run_round without touching its core
        globals()["type_benchmark"] = type("TBName", (), {"name": None})()
        globals()["rtl_top"] = args.rtl_top

    else:
        tb_enum_key = args.type_benchmark.upper()
        tb_enum = TypeBenchmark[tb_enum_key]

        spec_dict = get_benchmark_contents(
            tb_enum,
            TypeBenchmarkFile.SPEC,
            args.path_benchmark,
            args.filter_instance,
        )
        golden_tb_path_dict = get_benchmark_contents(
            tb_enum,
            TypeBenchmarkFile.TEST_PATH,
            args.path_benchmark,
            args.filter_instance,
        )
        golden_rtl_path_dict = get_benchmark_contents(
            tb_enum,
            TypeBenchmarkFile.GOLDEN_PATH,
            args.path_benchmark,
            args.filter_instance,
        )

        # Make enum visible to run_round without modifying its internals
        globals()["type_benchmark"] = tb_enum
        globals()["rtl_top"] = None

    llm = get_llm(
        model=args.model,
        cfg_path=args.key_cfg_path,
        max_token=args.max_token,
        provider=args.provider,
        temperature=args.temperature,
    )
    identifier_head = args.run_identifier
    n = args.n
    set_exp_setting(temperature=args.temperature, top_p=args.top_p)

    for i in range(n):
        print(f"Round {i+1}/{n}")
        args.run_identifier = f"{identifier_head}_{i}"
        run_round(args, llm, spec_dict, golden_tb_path_dict, golden_rtl_path_dict)


if __name__ == "__main__":
    main()
