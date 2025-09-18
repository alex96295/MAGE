import json
import os
import re
from typing import Dict, List, Tuple

from .bash_tools import CommandResult, run_bash_command
from .benchmark_read_helper import TypeBenchmark
from .log_utils import get_logger, set_log_dir

logger = get_logger(__name__)

BENIGN_STDERRS = [
    r"^\S+:\d+: sorry: constant selects in always_\* processes are not currently supported \(all bits will be included\)\.$"
]


def stderr_all_lines_benign(stderr: str) -> bool:
    return all(
        any(re.match(pattern, line) for pattern in BENIGN_STDERRS)
        for line in stderr.splitlines()
    )


def check_syntax(rtl_path: str, simulator: str) -> Tuple[bool, str]:
    rtl_path_lib = os.path.join(os.path.dirname(rtl_path), "rtl_lib.sv")
    match simulator:
        case "iverilog":
            cmd = f"oseda -2025.03 iverilog -t null -Wall -Winfloop -Wno-timescale -g2012 -o /dev/null {rtl_path_lib} {rtl_path}"
        case "questa":
            cmd = f"questa-2023.4 vlog {rtl_path_lib} {rtl_path} && questa-2023.4 vopt -permissive -suppress 3009 -suppress 8386 -error 7 +UVM_NO_RELNOTES -O5 TopTestbench -o TopTestbench_opt"
    is_pass, sim_output = run_bash_command(cmd, timeout=60)
    sim_output_obj = CommandResult.model_validate_json(sim_output)
    is_pass = (
        is_pass
        and "syntax error" not in sim_output_obj.stdout
        and (
            sim_output_obj.stderr == ""
            or stderr_all_lines_benign(sim_output_obj.stderr)
        )
    )
    logger.info(f"Syntax check is_pass: {is_pass}, \noutput: {sim_output}")
    return is_pass, sim_output


def sim_review_mismatch_cnt(stdout: str) -> int:
    mismatch_cnt = 0
    if "SIMULATION FAILED" in stdout:
        re_str = r"SIMULATION FAILED - (\d*) MISMATCHES DETECTED"
        m = re.search(re_str, stdout)
        assert m is not None, f"Failed to parse mismatch count from: {stdout}"
        mismatch_cnt = int(m.group(1))
    return mismatch_cnt


def sim_review(
    simulator: str,
    output_path_per_run: str,
    golden_rtl_path: str | None = None,
) -> Tuple[bool, int, str]:
    rtl_path = f"{output_path_per_run}/rtl.sv"
    rtl_path_lib = os.path.join(os.path.dirname(rtl_path), "rtl_lib.sv")
    vvp_name = f"{output_path_per_run}/sim_output.vvp"
    tb_path = f"{output_path_per_run}/tb.sv"
    if golden_rtl_path is None:
        golden_rtl_path = ""
    if os.path.isfile(vvp_name):
        os.remove(vvp_name)
    match simulator:
        case "iverilog":
            cmd = "oseda -2025.03 iverilog -Wall -Winfloop -Wno-timescale -g2012 -o {} {} {} {} {}; oseda -2025.03 vvp -n {}".format(
                vvp_name, tb_path, rtl_path_lib, rtl_path, golden_rtl_path, vvp_name
            )
        case "questa":
            cmd = 'questa-2023.4 vlog {} {} {} {} && questa-2023.4 vopt -permissive -suppress 3009 -suppress 8386 -error 7 +UVM_NO_RELNOTES -O5 TopTestbench -o TopTestbench_opt && questa-2023.4 vsim -c TopTestbench_opt -t 1ps -suppress 3009 -suppress 8386 -error 7 -cpppath /usr/bin/g++ -do "run -all; quit -f"'.format(
                rtl_path_lib, rtl_path, golden_rtl_path, tb_path
            )
    is_pass, sim_output = run_bash_command(cmd, timeout=60)
    sim_output_obj = CommandResult.model_validate_json(sim_output)
    is_pass = (
        is_pass
        and "SIMULATION PASSED" in sim_output_obj.stdout
        and (
            sim_output_obj.stderr == ""
            or stderr_all_lines_benign(sim_output_obj.stderr)
        )
    )
    mismatch_cnt = sim_review_mismatch_cnt(sim_output_obj.stdout)
    logger.info(
        f"Simulation is_pass: {is_pass}, mismatch_cnt: {mismatch_cnt}\noutput: {sim_output}"
    )
    assert isinstance(sim_output, str) and isinstance(is_pass, bool)
    return is_pass, mismatch_cnt, sim_output


class SimReviewer:
    def __init__(
        self,
        output_path_per_run: str,
        golden_rtl_path: str | None = None,
    ):
        self.output_path_per_run = output_path_per_run
        self.golden_rtl_path = golden_rtl_path

    def review(self, simulator: str) -> Tuple[bool, int, str]:
        return sim_review(
            simulator,
            self.output_path_per_run,
            self.golden_rtl_path,
        )


def sim_review_golden(
    rtl_path: str,
    task_id: str,
    benchmark_type: TypeBenchmark,
    benchmark_path: str,
    output_path_per_run: str,
) -> Tuple[bool, str]:

    if (
        benchmark_type == TypeBenchmark.VERILOG_EVAL_V2
        or benchmark_type == TypeBenchmark.VERILOG_EVAL_V1
    ):
        folder = (
            "dataset_code-complete-iccad2023"
            if benchmark_type == TypeBenchmark.VERILOG_EVAL_V1
            else "dataset_spec-to-rtl"
        )
        tb_path = f"{output_path_per_run}/tb.sv"
        ref_path = f"{benchmark_path}/{folder}/{task_id}_ref.sv"
        vvp_name = f"{output_path_per_run}/sim_golden.vvp"
        if os.path.isfile(vvp_name):
            os.remove(vvp_name)
        cmd = "oseda -2025.03 iverilog -Wall -Winfloop -Wno-timescale -g2012 -s tb -o {} {} {} {}; oseda -2025.03 vvp -n {}".format(
            vvp_name, tb_path, rtl_path, ref_path, vvp_name
        )
        is_pass, sim_output = run_bash_command(cmd, timeout=60)
        sim_output_obj = CommandResult.model_validate_json(sim_output)
        is_pass = (
            is_pass
            and "First mismatch occurred at time" not in sim_output_obj.stdout
            and (
                sim_output_obj.stderr == ""
                or stderr_all_lines_benign(sim_output_obj.stderr)
            )
        )
        logger.info(f"Golden simulation is_pass: {is_pass}, \noutput: {sim_output}")
        return is_pass, sim_output
    raise NotImplementedError  # Should not reach here


def sim_review_golden_benchmark(
    task_id: str,
    output_path: str,
    benchmark_type: TypeBenchmark,
    benchmark_path: str,
) -> Tuple[bool, str]:
    output_path_per_run = f"{output_path}/{benchmark_type.name}_{task_id}"
    rtl_path = f"{output_path_per_run}/rtl.sv"
    is_pass, sim_output = sim_review_golden(
        rtl_path, task_id, benchmark_type, benchmark_path, output_path_per_run
    )
    with open(f"{output_path_per_run}/sim_review_output.json", "w") as f:
        f.write(
            json.dumps(
                {"is_pass": is_pass, "sim_output": json.loads(sim_output)}, indent=4
            )
        )
    return (is_pass, sim_output)


def sim_review_golden_benchmark_batch(
    task_id_list: List[str],
    log_path: str,
    output_path: str,
    benchmark_type: TypeBenchmark,
    benchmark_path: str,
) -> Dict[str, Tuple[bool, str]]:
    ret: Dict[str, Tuple[bool, str]] = {}
    for task_id in task_id_list:
        set_log_dir(f"{log_path}/golden_review_{benchmark_type.name}_{task_id}")
        ret[task_id] = sim_review_golden_benchmark(
            task_id, output_path, benchmark_type, benchmark_path
        )
    return ret
