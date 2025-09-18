import os
import re
from typing import List, Optional, Tuple

from .bash_tools import CommandResult, run_bash_command
from .log_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Stderr whitelist (populate only if you identify recurring harmless lines).
# If left empty, any non-empty stderr is treated as potentially meaningful.
# ---------------------------------------------------------------------
BENIGN_STDERRS_SLANG: List[str] = [
    # example: r"^slang: note:.*$",
]


def _stderr_all_lines_benign(stderr: str) -> bool:
    """True if stderr is empty or each line matches a benign pattern."""
    if not stderr.strip():
        return True
    if not BENIGN_STDERRS_SLANG:
        return False
    return all(
        any(re.match(p, line) for p in BENIGN_STDERRS_SLANG)
        for line in stderr.splitlines()
    )


def _has_error_token(stdout: str, stderr: str) -> bool:
    """Lightweight error detector for slang diagnostics."""
    s = (stdout or "").lower()
    e = (stderr or "").lower()
    return (
        (" error " in f" {s} ")
        or (" error:" in s)
        or (" error " in f" {e} ")
        or (" error:" in e)
    )


def compile_slang(
    *,
    rtl_path: Optional[str] = None,
    mode: str = "elab",  # "parse" | "lint" | "elab"
    top: Optional[str] = None,
    std: Optional[str] = None,  # e.g. "1800-2017"
    timescale: Optional[str] = None,  # e.g. "1ns/1ns"
    extra_flags: Optional[List[str]] = None,  # e.g. ["--relax-enum-conversions", ...]
    compile_bin: str = "oseda -2025.03 slang",
    timeout_sec: int = 120,
) -> Tuple[bool, str]:
    """
    Run slang in the requested mode and return (is_pass, result_json_str).

    - Provide exactly one of rtl_path OR filelist.
    - mode="parse":       --parse-only
      mode="lint":        --lint-only
      mode="elab":        (no extra flag; performs parse+checks+elaboration)
    - Pass criteria: successful exit, no 'error' diagnostics, and benign/empty stderr.
    """

    rtl_path_lib = os.path.join(os.path.dirname(rtl_path), "rtl_lib.sv")

    args: List[str] = [compile_bin]
    args += [rtl_path_lib]
    args += [rtl_path]

    args += [f"--std={std}"] if std else [f"--std=1800-2017"]
    args += [f"--timescale={timescale}"] if timescale else [f"--timescale=1ns/1ns"]

    args += (
        ["--top", top]
        if top
        else logger.error("CompileReviewer Slang requires a top-level module")
    )

    if extra_flags:
        args += extra_flags

    mode = mode.strip().lower()
    if mode == "parse":
        args += ["--parse-only"]
    elif mode == "lint":
        args += ["--lint-only"]
    elif mode == "elab":
        # full elaboration: no extra mode flag
        pass
    else:
        raise ValueError(
            f"Unknown slang mode: {mode!r} (expected 'parse' | 'lint' | 'elab')"
        )

    cmd = " ".join(args)
    ok, result_json_str = run_bash_command(cmd, timeout=timeout_sec)

    # Parse the tool wrapper's JSON for robust pass criteria
    try:
        result = CommandResult.model_validate_json(result_json_str)
    except Exception:
        logger.info("Failed to parse slang output JSON; returning raw payload.")
        logger.info(f"Raw output: {result_json_str}")
        return False, result_json_str

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    is_pass = (
        ok
        and (not _has_error_token(stdout, stderr))
        and _stderr_all_lines_benign(stderr)
    )

    logger.info(f"[slang {mode}] cmd: {cmd}")
    logger.info(
        f"[slang {mode}] is_pass={is_pass} stdout_len={len(stdout)} stderr_len={len(stderr)}"
    )
    return is_pass, result_json_str


class CompileReviewer:
    """
    Non-LLM compile/elaboration checker (slang-backed).
    Mirrors SimReviewer style with a simple .review(mode) entry point.
    """

    def __init__(
        self,
        *,
        rtl_path: Optional[str] = None,
        filelist: Optional[str] = None,
        top: Optional[str] = None,
        timescale: Optional[str] = None,
        extra_flags: Optional[List[str]] = None,
        compile_bin: str = "oseda -2025.03 slang",
        timeout_sec: int = 120,
    ):
        assert (rtl_path is None) ^ (
            filelist is None
        ), "Provide exactly one of rtl_path OR filelist."
        self.rtl_path = rtl_path
        self.filelist = filelist
        self.top = top
        self.timescale = timescale
        self.extra_flags = list(extra_flags or [])
        self.compile_bin = compile_bin
        self.timeout_sec = timeout_sec

    def review(self, mode: str = "elab") -> Tuple[bool, str]:
        return compile_slang(
            rtl_path=self.rtl_path,
            filelist=self.filelist,
            mode=mode,
            top=self.top,
            timescale=self.timescale,
            extra_flags=self.extra_flags,
            compile_bin=self.compile_bin,
            timeout_sec=self.timeout_sec,
        )

    def review_parse(self) -> Tuple[bool, str]:
        return self.review(mode="parse")

    def review_lint(self) -> Tuple[bool, str]:
        return self.review(mode="lint")

    def review_elab(self) -> Tuple[bool, str]:
        return self.review(mode="elab")
