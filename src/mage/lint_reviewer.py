import json
import os
from typing import List, Optional, Sequence, Tuple

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from pydantic import BaseModel

from .bash_tools import CommandResult, run_bash_command
from .log_utils import get_logger
from .token_counter import TokenCounter, TokenCounterCached

logger = get_logger(__name__)

SYSTEM_PROMPT = r"""
You are a SystemVerilog style and lint fixer.
You will receive code that already passed simulation and a linter report.
Your job is to modify ONLY style/format/naming to satisfy the linter and formatter.

CRITICAL RULES:
- DO NOT change functional behavior or timing.
- DO NOT change ports, I/O directions, or interface semantics.
- DO NOT change constants or parameter default values that affect behavior.
- Prefer SystemVerilog-2017 constructs.
- Keep previous stylistic improvements unless they directly conflict with a lint rule.
"""

LINT_FIX_PROMPT = r"""
A linter (verible-verilog-lint) reported violations for the following sources.
Fix ONLY the reported issues while preserving behavior. Apply lowRISC RTL/DV style where applicable.

Linter output:
<lint_output>
{lint_output}
</lint_output>

You will also receive the current sources:

<rtl>
{rtl_code}
</rtl>

<tb>
{tb_code}
</tb>

Constraints:
- Do NOT alter behavior, timing, interfaces, or semantics.
- Make minimal changes necessary to satisfy lint rules.
- It is OK to rename signals/instances/types to satisfy naming rules, but preserve intent and connectivity.
- Avoid introducing new files or includes.

Return STRICT JSON ONLY with the following schema (no markdown, no extra keys):

{
  "reasoning": "brief explanation of the changes you applied to satisfy linter",
  "change_summary": ["bullet 1", "bullet 2", "..."],
  "rtl_lint_fixed": "FULL rewritten RTL source, SystemVerilog",
  "tb_lint_fixed": "FULL rewritten TB source, SystemVerilog"
}
"""

EXAMPLE_OUTPUT = {
    "reasoning": "Explained how lint errors were resolved without behavior change",
    "change_summary": [
        "Aligned instance ports; removed .*",
        "Renamed ports to *_i/*_o; ensured rst_ni and clk_i order",
        "Converted always @* to always_comb in TB helper block",
    ],
    "rtl_lint_fixed": "module ... endmodule",
    "tb_lint_fixed": "module tb; ... endmodule",
}


class LintFixOutput(BaseModel):
    reasoning: str
    change_summary: List[str]
    rtl_lint_fixed: str
    tb_lint_fixed: str


class LintReviewer:
    """
    Post-style pass agent focused on formatting and linting.

    Pipeline:
      1) verible-verilog-format (in-place) to normalize whitespace/indent/alignment.
      2) verible-verilog-lint check. If violations and enable_llm_fix=True:
         - feed lint output + current sources to the LLM for targeted fixes
         - write fixes, re-format, re-lint
         - up to lint_max_trials

    Supports verible's --autofix modes to reduce LLM overhead.

    Returns final RTL/TB contents (strings) so TopAgent decides what to persist.
    """

    def __init__(self, token_counter: TokenCounter):
        self.token_counter = token_counter
        self.history: List[ChatMessage] = []
        self.lint_max_trials = 3
        self.rag_chat_engine = None

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

    def reset(self) -> None:
        self.history = []

    def init_rag(
        self,
        persist_dir: str = "./.vector_storage/lint_reviewer",
        faiss_path: str = "./.faiss_storage/lint_reviewer_faiss.bin",
        docs: Optional[Sequence] = None,
    ) -> None:
        if not docs:
            logger.error(
                "LintReviewer No documents found to init RAG. Vector index will not be created."
            )
            return
        self.rag_chat_engine = self.token_counter.init_rag(
            persist_dir=persist_dir,
            faiss_path=faiss_path,
            top_k=2,
            memory_token_limit=1500,
            docs=docs,
        )
        logger.info("LintReviewer RAG initialized.")

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

    def _chat(self, messages: List[ChatMessage]) -> ChatResponse:
        logger.info(f"Lint reviewer input message count: {len(messages)}")
        resp, token_cnt = self.token_counter.count_chat(messages, self.rag_chat_engine)
        logger.info(f"Lint reviewer token count: {token_cnt}")
        logger.info(
            f"Lint reviewer raw response (first 500 chars): {resp.message.content[:500]}..."
        )
        return resp

    def _parse_fix(self, response: ChatResponse) -> Optional[LintFixOutput]:
        try:
            payload = json.loads(response.message.content, strict=False)
            return LintFixOutput(
                reasoning=payload["reasoning"],
                change_summary=list(payload["change_summary"]),
                rtl_lint_fixed=payload["rtl_lint_fixed"],
                tb_lint_fixed=payload["tb_lint_fixed"],
            )
        except Exception as e:
            logger.info(f"LintReviewer JSON parse error: {e}")
            return None

    def _messages_for_lint_fix(
        self, lint_output: str, rtl_code: str, tb_code: str
    ) -> List[ChatMessage]:
        msgs = [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(
                role=MessageRole.USER,
                content=LINT_FIX_PROMPT.format(
                    lint_output=lint_output,
                    rtl_code=rtl_code,
                    tb_code=tb_code,
                ),
            ),
        ]
        if (
            isinstance(self.token_counter, TokenCounterCached)
            and self.token_counter.enable_cache
        ):
            self.token_counter.add_cache_tag(msgs[-1])
        return msgs

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
        enable_llm_fix: bool = True,
    ) -> Tuple[str, str, bool, str, str, List[str], str]:
        rtl_path = os.path.join(output_dir_per_run, rtl_filename)
        tb_path = os.path.join(output_dir_per_run, tb_filename)

        # 1) Format
        fmt_ok, fmt_log = self._run_formatter([rtl_path, tb_path])
        stdout, stderr, rc = self._extract_stdout_stderr(fmt_log)
        logger.info(f"[Formatter] rc={rc}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")

        # 2) Lint
        lint_ok, lint_log = self._run_lint([rtl_path, tb_path])
        lint_stdout, lint_stderr, lint_rc = self._extract_stdout_stderr(lint_log)
        logger.info(
            f"[Lint] rc={lint_rc}\nSTDOUT:\n{lint_stdout}\nSTDERR:\n{lint_stderr}"
        )

        if lint_ok or not enable_llm_fix or self.lint_autofix_mode != "no":
            final_rtl, final_tb = self._read_files(rtl_path, tb_path)
            return (final_rtl, final_tb, lint_ok, lint_log, fmt_log, [], "")

        # 3) Lint-fix loop with LLM
        self.reset()
        self.token_counter.set_cur_tag(self.__class__.__name__)

        last_reasoning = ""
        last_summary: List[str] = []
        current_rtl, current_tb = self._read_files(rtl_path, tb_path)

        for trial in range(self.lint_max_trials):
            logger.info(f"[Lint-Fix] Trial {trial+1}/{self.lint_max_trials}")
            msgs = self._messages_for_lint_fix(
                lint_output=lint_stdout + ("\n" + lint_stderr if lint_stderr else ""),
                rtl_code=current_rtl,
                tb_code=current_tb,
            )
            resp = self._chat(msgs)
            parsed = self._parse_fix(resp)
            if (
                parsed
                and parsed.rtl_lint_fixed.strip()
                and parsed.tb_lint_fixed.strip()
            ):
                last_reasoning = parsed.reasoning
                last_summary = parsed.change_summary
                current_rtl = parsed.rtl_lint_fixed
                current_tb = parsed.tb_lint_fixed

                with open(rtl_path, "w") as f:
                    f.write(current_rtl)
                with open(tb_path, "w") as f:
                    f.write(current_tb)

            # Re-format + re-lint
            fmt_ok, fmt_log = self._run_formatter([rtl_path, tb_path])
            lint_ok, lint_log = self._run_lint([rtl_path, tb_path])
            if lint_ok:
                return (
                    current_rtl,
                    current_tb,
                    True,
                    lint_log,
                    fmt_log,
                    last_summary,
                    last_reasoning,
                )

        logger.warning(
            "Lint still failing after LLM fix loop; returning best-effort sources."
        )
        return (
            current_rtl,
            current_tb,
            False,
            lint_log,
            fmt_log,
            last_summary,
            last_reasoning,
        )

    def review(
        self,
        rtl_code: str,
        tb_code: str,
        output_dir_per_run: str,
        rtl_filename: str = "rtl.sv",
        tb_filename: str = "tb.sv",
        enable_llm_fix: bool = True,
    ) -> Tuple[str, str, bool, str, str, List[str], str]:
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
            enable_llm_fix=enable_llm_fix,
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
        self.extra_lint_args = list(extra_args) if extra_args else []

    def set_lint_autofix(
        self, mode: str = "no", output_file: Optional[str] = None
    ) -> None:
        """
        mode: one of {no, patch-interactive, patch, inplace-interactive, inplace, generate-waiver}
        output_file: path for --autofix_output_file when using 'patch' or 'generate-waiver'
        """
        self.lint_autofix_mode = mode
        self.lint_autofix_output_file = output_file
