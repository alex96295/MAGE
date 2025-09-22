import json
from typing import Dict, List, Optional, Sequence, Union

from llama_index.core import Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.generic_utils import prompt_to_messages
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.chat_engine.types import AgentChatResponse
from pydantic import BaseModel, Field

from .log_utils import get_logger
from .token_counter import TokenCounter, TokenCounterCached

logger = get_logger(__name__)

SYSTEM_PROMPT = r"""
You are an expert RTL Design Planner.
Goal: produce a structured, LLM-readable JSON plan BEFORE any RTL is written.
You DO NOT write SystemVerilog here. You only plan reuse and consultation,
following the planning goals below.

Planning goals:
1) REUSE: identify concrete, primitive submodules likely to be instantiated in
the final design. This are submodules that you wish, as a design planner, to
have in the design and that another independent agent will try to fetch from a
design library. This phase of design planning is the high-level equivalent of
reasoning with the block diagram of the module, based on the specification.
2) CONSULT: identify similar designs (even with different protocols) to guide
naming, parameters, and structure. These submodules are not for reuse
(instantiation) in the design to be generated, but merely for consultancy
goals. It shows you how similar modules in intent/protocol have been designed
in an existing design library.

Rules:
- Base your plan ONLY on the input_spec and (optionally) a provided interface and/or testbench sketch.
- The plan is a JSON object with exactly two top-level keys: "reuse" and "consult".
- Under each, create keys like "module1", "module2", ... Each value has:
  - "description": short natural language of what to reuse/consult
  - "keywords": list of 8-10 short tokens
  - "protocols": list of protocols or 'generic'
  - "reasoning": a short paragraph explaining why this is relevant

Do NOT include any SystemVerilog code. Do NOT add extra top-level fields.
"""

PLANNING_PROMPT = r"""
You will plan for the following design:

<input_spec>
{input_spec}
</input_spec>

Now: produce ONLY the JSON plan described.
"""

ORDER_PROMPT = r"""
Output constraints (mandatory):
- Return ONLY valid JSON (no markdown fences, no commentary).
- Top-level keys MUST be exactly: "reuse" and "consult".
- Under each, create at least one entry if applicable (module1, module2, ...). If none, use {}.
- Example shape:

{
  "reuse": {
    "module1": {
      "description": "credit-based flow counter for backpressure",
      "keywords": ["credit", "flow", "counter"],
      "protocols": ["generic"],
      "reasoning": "Needed to track send-side credits..."
    }
  },
  "consult": {
    "module1": {
      "description": "demux structure implemented for AXI",
      "keywords": ["demux", "arbiter", "AXI"],
      "protocols": ["AXI4"],
      "reasoning": "Similar routing behavior, aligns naming and port style..."
    }
  }
}
"""


class PlannerOutput(BaseModel):
    reuse: Dict[str, Dict] = Field(default_factory=dict)
    consult: Dict[str, Dict] = Field(default_factory=dict)


class DesignPlanner:
    """
    LLM-backed planning front-end that proposes what to REUSE and what to CONSULT
    from the library, but does not query the library directly.
    """

    def __init__(self, token_counter: TokenCounter):
        self.token_counter = token_counter
        self.history: List[ChatMessage] = []
        self.rag_chat_engine: Optional[ContextChatEngine] = None
        self.generated_tb: Optional[str] = None
        self.generated_if: Optional[str] = None

    def reset(self) -> None:
        self.history = []
        self.generated_tb = None
        self.generated_if = None

    def init_rag(
        self,
        persist_dir: str = "./.vector_storage/design_planner",
        faiss_path: str = "./.faiss_storage/design_planner_faiss.bin",
        docs: Sequence[Document] = None,
        embed_model: BaseEmbedding = None,
        memory_token_limit: int = 1500,
    ) -> None:
        """
        Optional: allow lightweight RAG (e.g., style guides or methodology notes).
        Not used for deterministic library consulting; that's handled in lib_consultant.py.
        """
        if not docs:
            logger.info("DesignPlanner: no docs provided for RAG. Skipping.")
            return
        self.rag_chat_engine = self.token_counter.init_rag(
            persist_dir=persist_dir,
            faiss_path=faiss_path,
            top_k=2,
            memory_token_limit=memory_token_limit,
            docs=docs,
            embed_model=embed_model,
        )
        logger.info("DesignPlanner RAG initialized.")

    def generate(
        self, messages: List[ChatMessage]
    ) -> Union[ChatResponse, AgentChatResponse]:
        resp, token_cnt = self.token_counter.count_chat(
            messages, rag_chat_engine=self.rag_chat_engine
        )
        logger.info(f"DesignPlanner token count: {token_cnt}")
        return resp

    def parse_output(self, response: ChatResponse) -> PlannerOutput:
        try:
            obj = json.loads(response.message.content, strict=False)
            return PlannerOutput(**obj)
        except Exception as e:
            logger.warning(f"[DesignPlanner] JSON parse failed: {e}")
            return PlannerOutput()

    def chat(
        self,
        input_spec: str,
        enable_cache: bool = False,
        max_trials: int = 5,
    ) -> str:
        """
        Produce a planning JSON (string) and a parsed PlannerOutput.
        Consistent with RTLGenerator: fresh history, cache/tag handling, retry loop.
        """

        if isinstance(self.token_counter, TokenCounterCached):
            self.token_counter.set_enable_cache(enable_cache)
        try:
            self.token_counter.set_cur_tag(self.__class__.__name__)
        except Exception:
            pass

        self.history = []

        base_messages: List[ChatMessage] = [
            ChatMessage(content=SYSTEM_PROMPT, role=MessageRole.SYSTEM),
            ChatMessage(
                content=PLANNING_PROMPT.format(
                    input_spec=input_spec,
                ),
                role=MessageRole.USER,
            ),
        ]
        self.history.extend(base_messages)

        parsed: Optional[PlannerOutput] = None
        plan_json_str: str = ""

        for _ in range(max_trials):
            response = self.generate(
                self.history
                + [ChatMessage(content=ORDER_PROMPT, role=MessageRole.USER)]
            )
            if isinstance(response, ChatResponse):
                raw_text = response.message.content
                message = response.message
            elif isinstance(response, AgentChatResponse):
                raw_text = response.response
                message = prompt_to_messages(raw_text)
            else:
                raise TypeError(f"Unexpected response type: {type(response)}")

            # Try to parse
            try:
                obj = json.loads(raw_text, strict=False)
                parsed_candidate = PlannerOutput(**obj)
                # success
                parsed = parsed_candidate
                payload = (
                    parsed.model_dump()
                    if hasattr(parsed, "model_dump")
                    else parsed.dict()
                )
                plan_json_str = json.dumps(payload, indent=2, ensure_ascii=False)
                # keep last successful assistant message in history (debug parity with RTL)
                self.history.append(message)
                break
            except Exception:
                self.history.append(message)
                continue

        # If all attempts failed, return an empty but valid object
        if parsed is None:
            parsed = PlannerOutput()
            payload = (
                parsed.model_dump() if hasattr(parsed, "model_dump") else parsed.dict()
            )
            plan_json_str = json.dumps(payload, indent=2, ensure_ascii=False)

        return plan_json_str
