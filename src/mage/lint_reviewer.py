import os
from typing import List, Optional, Tuple

from .bash_tools import CommandResult, run_bash_command
from .log_utils import get_logger

logger = get_logger(__name__)


class LintReviewer:
    """
    Post-style pass agent focused on formatting and linting using tooling only.

    Pipeline:
      1) verible-verilog-format (in-place) to normalize whitespace/indent/alignment.
      2) verible-verilog-lint check.

    Supports verible's --autofix modes.

    Returns final RTL/TB contents (strings) so TopAgent decides what to persist.
    """

    def __init__(self):
        # Formatter defaults
        self.format_args: List[str] = [
            "--column_limit=100",
            "--wrap_spaces=2",
            "--indentation_spaces=2",
            "--line_break_penalty=2",
            "--assignment_statement_alignment=align",
            "--module_net_variable_alignment=flush-left",
            "--named_parameter_alignment=align",
            "--named_port_alignment=align",
            "--port_declarations_alignment=preserve",
        ]

        # Lint config
        self.rules_config_file: Optional[str] = None
        self.waiver_files: List[str] = []
        self.lint_args: List[str] = []

        # Autofix
        self.lint_autofix_mode: str = "no"
        self.lint_autofix_output_file: Optional[str] = None

    def _run_formatter(self, paths: List[str]) -> Tuple[bool, str]:
        ok_all = True
        combined = []
        for p in paths:
            args = ["oseda -2025.03 verible-format", "--inplace"]
            args += self.format_args
            args.append(p)
            cmd = " ".join(args)
            is_ok, out = run_bash_command(cmd, timeout=60)
            combined.append(out)
            ok_all = ok_all and is_ok
        return ok_all, "\n".join(combined)

    def _run_lint(self, paths: List[str]) -> Tuple[bool, str]:
        args = [
            "oseda -2025.03 verible-lint",
            f"--autofix={self.lint_autofix_mode}",
            "--parse_fatal",
            "--lint_fatal",
            "--error_limit=0",
        ]
        if self.lint_autofix_output_file and self.lint_autofix_mode in (
            "patch",
            "patch-interactive",
            "generate-waiver",
        ):
            args.append(f"--autofix_output_file={self.lint_autofix_output_file}")
        if self.rules_config_file:
            args.append(f"--rules_config={self.rules_config_file}")
        for wf in self.waiver_files:
            args.append(f"--waiver_files={wf}")
        args += self.lint_args
        args += paths
        cmd = " ".join(args)
        is_ok, out = run_bash_command(cmd, timeout=120)
        return is_ok, out

    @staticmethod
    def _extract_stdout_stderr(result_json_str: str) -> Tuple[str, str, int]:
        try:
            obj = CommandResult.model_validate_json(result_json_str)
            rc = getattr(obj, "returncode", None)
            if rc is None:
                rc = getattr(obj, "exit_code", 1)
            return obj.stdout or "", obj.stderr or "", int(rc)
        except Exception:
            return result_json_str, "", 1

    @staticmethod
    def _read_files(rtl_path: str, tb_path: str) -> Tuple[str, str]:
        with open(rtl_path, "r") as f:
            rtl = f.read()
        with open(tb_path, "r") as f:
            tb = f.read()
        return rtl, tb

    def run_on_paths(
        self,
        output_dir_per_run: str,
        rtl_filename: str = "rtl.sv",
        tb_filename: str = "tb.sv",
    ) -> Tuple[str, str, bool, str, str, List[str], str]:
        """
        Run formatter and linter on provided RTL/TB file paths.

        Returns:
          (final_rtl_str, final_tb_str, lint_ok, lint_log, fmt_log, change_summary, reasoning)

        'change_summary' and 'reasoning' are kept for API compatibility but unused here.
        """
        rtl_path = os.path.join(output_dir_per_run, rtl_filename)
        tb_path = os.path.join(output_dir_per_run, tb_filename)

        # 1) Format
        fmt_ok, fmt_log = self._run_formatter([rtl_path, tb_path])
        stdout, stderr, rc = self._extract_stdout_stderr(fmt_log)
        logger.info(f"[Formatter] rc={rc}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")

        # 2) Lint (optionally with verible autofix modes)
        lint_ok, lint_log = self._run_lint([rtl_path, tb_path])
        lint_stdout, lint_stderr, lint_rc = self._extract_stdout_stderr(lint_log)
        logger.info(
            f"[Lint] rc={lint_rc}\nSTDOUT:\n{lint_stdout}\nSTDERR:\n{lint_stderr}"
        )

        final_rtl, final_tb = self._read_files(rtl_path, tb_path)
        # No LLM loop; we strictly rely on tool outcomes.
        return (final_rtl, final_tb, lint_ok, lint_log, fmt_log, [], "")

    def review(
        self,
        rtl_code: str,
        tb_code: str,
        output_dir_per_run: str,
        rtl_filename: str = "rtl.sv",
        tb_filename: str = "tb.sv",
    ) -> Tuple[str, str, bool, str, str, List[str], str]:
        """
        Write sources to disk, then run formatter and linter.
        """
        rtl_path = os.path.join(output_dir_per_run, rtl_filename)
        tb_path = os.path.join(output_dir_per_run, tb_filename)
        os.makedirs(output_dir_per_run, exist_ok=True)
        with open(rtl_path, "w") as f:
            f.write(rtl_code)
        with open(tb_path, "w") as f:
            f.write(tb_code)
        return self.run_on_paths(
            output_dir_per_run=output_dir_per_run,
            rtl_filename=rtl_filename,
            tb_filename=tb_filename,
        )

    def set_formatter_options(self, args: Optional[List[str]] = None) -> None:
        if args is not None:
            self.format_args = list(args)

    def set_lint_options(
        self,
        rules_config_file: Optional[str] = None,
        waiver_files: Optional[List[str]] = None,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        self.rules_config_file = rules_config_file
        self.waiver_files = list(waiver_files) if waiver_files else []
        self.lint_args = list(extra_args) if extra_args else []

    def set_lint_autofix(
        self, mode: str = "no", output_file: Optional[str] = None
    ) -> None:
        """
        mode: one of {no, patch-interactive, patch, inplace-interactive, inplace, generate-waiver}
        output_file: path for --autofix_output_file when using 'patch' or 'generate-waiver'
        """
        self.lint_autofix_mode = mode
        self.lint_autofix_output_file = output_file
