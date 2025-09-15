# style_reviewer.py
import json
from typing import List, Optional, Sequence, Tuple

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.chat_engine import ContextChatEngine
from pydantic import BaseModel

from .log_utils import get_logger
from .token_counter import TokenCounter, TokenCounterCached

logger = get_logger(__name__)


# --------------------------
# Prompts
# --------------------------

SYSTEM_PROMPT = r"""
You are a senior hardware style reviewer. The RTL and TB provided to you have
ALREADY PASSED syntax and simulation. Your sole task is to improve CODE STYLE,
FORMAT, and NAMING according to the lowRISC Comportable style guides.

CRITICAL RULES:
- DO NOT change functional behavior or timing.
- DO NOT change ports, I/O directions, or interface semantics.
- DO NOT change constants or parameter default values that affect behavior.
- Assume other agents guaranteed correctness; you must only make stylistic edits.
- Prefer SystemVerilog-2017 constructs.
"""

STYLE_CHECKLIST_PROMPT = r"""
Style Review Scope (lowRISC RTL style; apply where relevant):
1) Naming & Hierarchy
   - Signals, ports, instances, and types follow lowRISC naming:
     * lower_snake_case for signals, instances, variables, packages.
     * UpperCamelCase for parameters and enum values (unless ALL_CAPS constants).
   - Ports: add suffix `_i` (inputs), `_o` (outputs), `_io` (bidirectional).
   - Active-low signals end with `_n` (e.g., rst_ni).
   - Clock/reset ports listed first: `clk_i`, `rst_ni`.
   - Sequential logic naming:
     * `_d` for next-state signals (combinational input to flop).
     * `_q` for flop outputs.
     * `_q2`, `_q3` for pipelined versions (2nd stage, 3rd stage, etc.).
   - FSM states declared as enums:
     * Enum type name suffixed `_e` (e.g., state_e).
     * Enum values in UpperCamelCase (e.g., StIdle, StInit).
   - Module instance names:
     * Always lower_snake_case.
     * Prefix with `i_` to indicate instantiation (e.g., `i_fifo`, `i_arbiter`).
   - Hierarchical consistency: a signal connected across modules keeps the same name at all hierarchy levels.
2) Language & Constructs
   - Use logic over reg/wire (wire only when truly needed, e.g., inout).
   - Use always_ff/always_comb/always_latch appropriately.
   - Non-blocking (<=) in sequential; blocking (=) in combinational.
   - No tabs; <=100 chars target; whitespace around operators; compact array dims.
   - Prefer unique case + default; avoid full_case/parallel_case/casex; allow case inside/casez when justified.
   - Explicit literal widths; avoid boolean-context multi-bit signals.
   - No X assignments to model don't-cares; suggest/insert ASSERT/ASSERT_KNOWN where helpful without breaking sim.
3) Modules & Instances
   - Verilog-2001 full port declarations; clock/reset first.
   - Named port connections; no .*; tabular alignment of instance ports.
   - Named parameters on instantiation.
4) Constants & Types
   - Use localparam/parameter with explicit types; encourage package constants for globals.
   - Typedef enums with _e; enumerants typically UpperCamelCase or ALL_CAPS as per guide; explicit enum base type.
5) Formatting & Comments
   - C++ style // comments; section headers; begin/end placement rules; spacing for labels/commas/keywords.
   - Remove trailing whitespace; use spaces, not tabs; consistent indent (2 spaces).
6) FSMs
   - Two-process style (comb/seq), defaults before case, clear next-state assignment pattern.
7) Assertions (non-functional)
   - Encourage `ASSERT_KNOWN` on outputs and critical controls where safe; include only if it does not affect behavior.
"""

TB_CHECKLIST_PROMPT = r"""
Testbench Styling:
- First, detect whether the TB uses UVM (e.g., uvm_* classes, factory, phases).
- If **UVM is used**, apply lowRISC DV/UVM style guidance (naming, factory usage, component/sequence class naming,
  configuration patterns, objection handling, phase methods, macro invocation spacing) *without* changing functionality,
  as well as style the TB using the **RTL style guide** only (naming, always_comb, formatting, module instantiation).
- If **UVM is NOT used**, disregard any DV/UVM rule and only style the TB using the **RTL style guide** only (naming, always_comb, formatting, module instantiation).
- In both cases: do not add or remove TB functionality; do not change stimulus semantics or checkers' logic. If a style change implies a functionality change, drop the style change.
"""

IO_PROMPT = r"""
You will receive:
- <rtl> ... </rtl> : the synthesizable RTL (SystemVerilog).
- <tb>  ... </tb>  : the passing testbench.

Rewrite both with STYLE-ONLY changes according to the scopes above.

DO NOT:
- Change behavior, port lists, parameters, or timing.
- Introduce/require new files or includes.

Return a strict JSON object with this schema (no extra keys, no markdown):

{
  "reasoning": "brief explanation of style changes and key checks you applied",
  "is_uvm_tb": true|false,
  "change_summary": ["bullet 1", "bullet 2", "..."],
  "rtl_styled": "FULL rewritten RTL source, SystemVerilog",
  "tb_styled": "FULL rewritten TB source, SystemVerilog"
}
"""

EXAMPLE_OUTPUT = {
    "reasoning": "Short explanation of applied style improvements",
    "is_uvm_tb": False,
    "change_summary": [
        "Renamed ports to *_i/*_o, placed clk_i/rst_ni first.",
        "Converted always @* to always_comb; blocking in comb, non-blocking in seq.",
        "Reordered module header, aligned instance ports.",
    ],
    "rtl_styled": "module ... endmodule",
    "tb_styled": "module tb; ... endmodule",
}


class StyleReviewOutput(BaseModel):
    reasoning: str
    is_uvm_tb: bool
    change_summary: List[str]
    rtl_styled: str
    tb_styled: str


class StyleReviewer:
    """
    Final-pass style agent. Only runs after TB/RTL pass syntax and simulation.
    It preserves functionality and focuses solely on style per lowRISC guides.
    """

    def __init__(self, token_counter: TokenCounter):
        self.token_counter = token_counter
        self.history: List[ChatMessage] = []
        self.max_trials = 3  # style edits should converge quickly
        self.rag_chat_engine: Optional[ContextChatEngine] = None

    def reset(self) -> None:
        self.history = []

    def init_rag(
        self,
        persist_dir: str = "./.vector_storage/style_reviewer",
        faiss_path: str = "./.faiss_storage/style_reviewer_faiss.bin",
        docs: Optional[Sequence] = None,
        embed_model: BaseEmbedding = None,
    ) -> None:
        """
        Build/load a vector index from docs and create a retriever/chat engine.
        We keep arguments consistent with other agents so you can feed the
        lowRISC style docs into this agent's RAG.
        """
        if not docs:
            logger.error(
                "StyleReviewer No documents found to init RAG. Vector index will not be created."
            )
            return

        # Reuse token_counter helper to stand up RAG the same way as other agents.
        self.rag_chat_engine = self.token_counter.init_rag(
            persist_dir=persist_dir,
            faiss_path=faiss_path,
            top_k=2,
            memory_token_limit=1500,
            docs=docs,
            embed_model=embed_model,
        )
        logger.info("StyleReviewer RAG initialized.")

    def _messages_for_review(self, rtl_code: str, tb_code: str) -> List[ChatMessage]:
        msgs = [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(
                role=MessageRole.USER,
                content=STYLE_CHECKLIST_PROMPT
                + "\n"
                + TB_CHECKLIST_PROMPT
                + "\n"
                + IO_PROMPT,
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=f"<rtl>\n{rtl_code}\n</rtl>\n<tb>\n{tb_code}\n</tb>",
            ),
        ]
        # If caching is enabled, tag last user message for cache affinity.
        if (
            isinstance(self.token_counter, TokenCounterCached)
            and self.token_counter.enable_cache
        ):
            self.token_counter.add_cache_tag(msgs[-1])
        return msgs

    def _chat(self, messages: List[ChatMessage]) -> ChatResponse:
        logger.info(f"Style reviewer input message count: {len(messages)}")
        resp, token_cnt = self.token_counter.count_chat(messages, self.rag_chat_engine)
        logger.info(f"Style reviewer token count: {token_cnt}")
        logger.info(f"Style reviewer raw response: {resp.message.content[:500]}...")
        return resp

    def _parse_output(self, response: ChatResponse) -> Optional[StyleReviewOutput]:
        try:
            payload = json.loads(response.message.content, strict=False)
            return StyleReviewOutput(
                reasoning=payload["reasoning"],
                is_uvm_tb=payload["is_uvm_tb"],
                change_summary=list(payload["change_summary"]),
                rtl_styled=payload["rtl_styled"],
                tb_styled=payload["tb_styled"],
            )
        except Exception as e:
            logger.info(f"StyleReviewer JSON parse error: {e}")
            return None

    def review(
        self,
        rtl_code: str,
        tb_code: str,
        enable_cache: bool = False,
    ) -> Tuple[str, str, List[str], str]:
        """
        Perform a style-only rewrite for RTL and TB.

        Returns:
            (rtl_styled, tb_styled, change_summary, reasoning)

        Notes:
            - No simulation or syntax checks here; this agent runs AFTER everything passes.
            - If the model returns invalid JSON, we retry up to max_trials.
        """
        if isinstance(self.token_counter, TokenCounterCached):
            self.token_counter.set_enable_cache(enable_cache)

        self.reset()
        self.token_counter.set_cur_tag(self.__class__.__name__)

        messages = self._messages_for_review(rtl_code, tb_code)

        last_reasoning = ""
        last_summary: List[str] = []
        last_rtl = rtl_code
        last_tb = tb_code

        for _ in range(self.max_trials):
            resp = self._chat(messages)
            parsed = self._parse_output(resp)
            if (
                parsed is not None
                and parsed.rtl_styled.strip()
                and parsed.tb_styled.strip()
            ):
                return (
                    parsed.rtl_styled,
                    parsed.tb_styled,
                    parsed.change_summary,
                    parsed.reasoning,
                )

            # Fallback: append a minimal correction hint and retry
            last_reasoning = "Model returned invalid JSON or empty fields; retrying with stricter formatting."
            last_summary = ["Enforced strict JSON schema; no functional changes."]
            messages.append(
                ChatMessage(
                    role=MessageRole.USER,
                    content=(
                        "Your previous response was not valid JSON or had empty fields. "
                        "Please return STRICT JSON ONLY, matching the exact schema keys and types. "
                        "Do not include markdown. Do not omit any required fields."
                    ),
                )
            )

        # If all retries fail, return originals with a minimal note.
        logger.warning(
            "StyleReviewer failed to get valid JSON after retries. Returning originals."
        )
        return (last_rtl, last_tb, last_summary, last_reasoning)
