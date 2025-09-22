import json
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

from llama_index.core import Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.generic_utils import prompt_to_messages
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.chat_engine.types import AgentChatResponse
from pydantic import BaseModel

from .compile_reviewer import compile_slang
from .log_utils import get_logger
from .prompts import (
    FAILED_TRIAL_PROMPT,
    ORDER_PROMPT,
    RTL_4_SHOT_EXAMPLES,
    SV_LANGUAGE_DIRECTIVES_PROMPT,
)
from .sim_reviewer import check_syntax
from .token_counter import TokenCounter, TokenCounterCached
from .utils import _append_rtl_libs, _collect_rtl_libs, _safe_json_loads, add_lineno

logger = get_logger(__name__)


SYSTEM_PROMPT = r"""
You are an expert in RTL design.
You can always write SystemVerilog code with
no syntax errors and always reach correct functionality.

All generated SystemVerilog code must strictly conform to the IEEE 1800-2017
SystemVerilog Language Reference Manual (LRM) and follow the best practices
described in "Verilog and SystemVerilog Gotchas: 101 Common Coding Errors and
How to Avoid Them", by Stuart Sutherland and Don Mills.

{sv_language_directives_prompt}
"""

GENERATION_PROMPT = r"""
Write a module in SystemVerilog RTL language regarding to the given natural language specification.
Understand the requirements above and give reasoning steps in natural language to achieve it.
In addition, try to give advice to avoid syntax error.
An SystemVerilog RTL module always starts with a line starting with the keyword 'module' followed by the module name.
It ends with the keyword 'endmodule'.

[Hints]:
For implementing kmap (Karnaugh map), you need to think step by step.
Carefully example how the kmap in input_spec specifies the order of the inputs.
Note that x[i] in x[N:1] means x[i-1] in x[N-1:0].
Then find the inputs corresponding to output=1, 0, and don't-care for each case.

Note in Verilog, for a signal "logic x[M:N]" where M > N, you CANNOT reversely select bits from it like x[1:2];
Instead, you should use concatations like {{x[1], x[2]}}.

The module interface should EXACTLY MATCH module_interface if given.
Otherwise, should EXACTLY MATCH with the description in input_spec.
(Including the module name, input/output ports names, and their types)


{examples_prompt}
<input_spec>
{input_spec}
</input_spec>
"""

EXTRA_ORDER_PROMPT = r"""
Other requirements:
1. Don't use state_t to define the parameter. Use `localparam` or Use 'reg' or 'logic' for signals as registers or Flip-Flops.
2. Declare all ports and signals as logic.
3. Not all the sequential logic need to be reset to 0 when reset is asserted,
    but these without-reset logic should be initialized to a known value with an initial block instead of being X.
4. For combinational logic with an always block do not explicitly specify the sensitivity list; instead use always @(*).
5. NEVER USE 'inside' operator in RTL code. Code like 'state inside {STATE_B, STATE_C, STATE_D}' should NOT be used.
6. Never USE 'unique' or 'unique0' keywords in RTL code. Code like 'unique case' should NOT be used.
7. When instantiating reused modules, match parameter names and port lists EXACTLY as in lib_consultant_json (names, directions, widths). Do not rename their ports or parameters.
8. Output exactly one module: the primary design you are generating. Do not include reused module source; only instantiate them. The build will append their declarations.
9. Important: if previous files included appended reused modules, ignore them as they will be re-appended by the build. Only modify the primary module, that is your focus.
"""
# Some prompts above come from:
# @misc{ho2024verilogcoderautonomousverilogcoding,
#       title={VerilogCoder: Autonomous Verilog Coding Agents with Graph-based Planning and Abstract Syntax Tree (AST)-based Waveform Tracing Tool},
#       author={Chia-Tung Ho and Haoxing Ren and Brucek Khailany},
#       year={2024},
#       eprint={2408.08927},
#       archivePrefix={arXiv},
#       primaryClass={cs.AI},
#       url={https://arxiv.org/abs/2408.08927},
# }

IF_PROMPT = r"""
The module interface is given below:
<module_interface>
{module_interface}
</module_interface>
"""

TB_PROMPT = r"""
Another agent has generated a testbench regarding the given input_spec:
<testbench>
{testbench}
</testbench>
"""

FORMAT_ERROR_PROMPT = r"""
The error below has been reported by the tools:
<format_error>
{format_error}
</format_error>

Below is the current PRIMARY module with line numbers. (Reused library module declarations are appended automatically by the build; do not edit or duplicate them.)
To understand the error message better, we offered a version of generated main module with line number:
<module_with_lineno>
{module_with_lineno}
</module_with_lineno>
"""

EXAMPLE_OUTPUT = {
    "reasoning": "All reasoning steps and advices to avoid syntax error",
    "module": "Pure SystemVerilog code, a complete module",
}

INTEGRATION_GUIDANCE_PROMPT = r"""
You are given two structured inputs from an independent agent based on the
input spec defined above for the module to be designed. These strctured inputs
represent the reasoning of the independent agent in understanding the
high-level building blocks (submodules) of the design, and the effective
submodules present in the available RTL library.

<planner_json>
{planner_json}
</planner_json>

<lib_consultant_json>
{lib_consultant_json}
</lib_consultant_json>

How to use them:
1) Reuse candidates are under lib_consultant_json.reuse.*.library_json.
   - Use the interface (parameters + ports) exactly as listed to INSTANTIATE those modules inside your new design.
   - Do not alter reused module port names, directions, or parameter names.
   - Define connetion signals with the right type and width for proper connection between the instantiated modules and the surrounding logic.
2) Do not include the reused module source body in your own "module" output.
   - The build system will append the correct reused module declarations/definitions after your new module.
   - Your job: instantiate them correctly (parameters/ports, widths, resets, naming).
3) If no confident match exists (library_json == null), implement the needed logic yourself.
4) Prefer naming conventions / structure suggested by the planner_json when reasonable.
5) If a preliminary interface is provided, it takes precedence over any other suggestion.
6) Output only one complete SystemVerilog module for the requested design, not testbench code.

When instantiating reused modules:
- Map your top-level interface signals to the reused module ports clearly and consistently.
- Declare any internal wires/regs (logic) needed to connect to those instances.
- Avoid `unique/unique0`, avoid `inside`, and follow the rest of the RTL rules above.

You will ONLY author the primary design module.
Reused library modules are appended automatically by the build system.

Rules:
- Instantiate reused modules using the interface in <lib_consultant_json>.
- Do NOT include reused module source bodies in your output.
- If you are shown a previous file that contains the primary module PLUS appended reused module declarations, IGNORE those appended modules. ONLY modify and output the primary module.
- Your response must contain exactly one complete SystemVerilog module: the primary design.
"""


class RTLOutputFormat(BaseModel):
    reasoning: str
    module: str


class RTLGenerator:
    def __init__(
        self,
        token_counter: TokenCounter,
    ):
        self.token_counter = token_counter
        self.generated_tb: str | None = None
        self.generated_if: str | None = None
        self.failed_trial: List[ChatMessage] = []
        self.history: List[ChatMessage] = []
        self.max_trials = 5
        self.enable_cache = (False,)
        self.rag_chat_engine: Optional[ContextChatEngine] = None
        self.planner_json_str: Optional[str] = None
        self.lib_consult_json_str: Optional[str] = None

    def reset(self):
        self.history = []

    def init_rag(
        self,
        persist_dir: str = "./.vector_storage/tb_gen",
        faiss_path: str = "./.faiss_storage/tb_gen_faiss.bin",
        docs: Sequence[Document] = None,
        embed_model: BaseEmbedding = None,
        memory_token_limit: int = 1500,
    ) -> None:
        """
        Build/load a vector index from docs, create a retriever and a chat engine.
        """

        if not docs:
            logger.error(
                "RTLGenerator No documents found to init RAG. "
                "Vector index will not be created."
            )
            return

        self.rag_chat_engine = self.token_counter.init_rag(
            persist_dir=persist_dir,
            faiss_path=faiss_path,
            top_k=2,
            memory_token_limit=memory_token_limit,
            docs=docs,
            embed_model=embed_model,
        )

        logger.info("RTLGenerator RAG initialized.")

    def set_failed_trial(
        self, failed_sim_log: str, previous_code: str, previous_tb: str
    ) -> None:
        cur_failed_trial = FAILED_TRIAL_PROMPT.format(
            failed_sim_log=failed_sim_log,
            previous_code=add_lineno(previous_code),
            previous_tb=add_lineno(previous_tb),
        )
        self.failed_trial.append(
            ChatMessage(content=cur_failed_trial, role=MessageRole.USER)
        )

    def generate(
        self, messages: List[ChatMessage]
    ) -> Union[ChatResponse, AgentChatResponse]:
        logger.info(f"RTL generator input message: {messages}")
        resp, token_cnt = self.token_counter.count_chat(
            messages, rag_chat_engine=self.rag_chat_engine
        )
        logger.info(f"Token count: {token_cnt}")
        if isinstance(resp, ChatResponse):
            raw_text = resp.message.content
        elif isinstance(resp, AgentChatResponse):
            raw_text = resp.response
        else:
            raise TypeError(f"Unexpected response type: {type(resp)}")
        logger.info(f"{raw_text}")
        return resp

    def batch_generate(
        self, messages_list: List[List[ChatMessage]]
    ) -> List[Union[ChatResponse, AgentChatResponse]]:
        resp_token_cnt_list = self.token_counter.count_chat_batch(messages_list)
        responses = []
        for i, ((resp, token_cnt), _) in enumerate(
            zip(resp_token_cnt_list, messages_list)
        ):
            logger.info(f"Message {i+1} token count: {token_cnt}")
            responses.append(resp)
        return responses

    def get_init_prompt_messages(self, input_spec: str) -> List[ChatMessage]:
        ret = [
            ChatMessage(
                content=SYSTEM_PROMPT.format(
                    sv_language_directives_prompt=SV_LANGUAGE_DIRECTIVES_PROMPT
                ),
                role=MessageRole.SYSTEM,
            ),
            ChatMessage(
                content=GENERATION_PROMPT.format(
                    input_spec=input_spec, examples_prompt=RTL_4_SHOT_EXAMPLES
                ),
                role=MessageRole.USER,
            ),
        ]
        if self.generated_tb:
            ret.append(
                ChatMessage(
                    content=TB_PROMPT.format(testbench=self.generated_tb),
                    role=MessageRole.USER,
                )
            )
        if self.failed_trial:
            ret.extend(self.failed_trial)
        if self.generated_if:
            ret.append(
                ChatMessage(
                    content=IF_PROMPT.format(module_interface=self.generated_if),
                    role=MessageRole.USER,
                )
            )

        if self.planner_json_str or self.lib_consult_json_str:
            ret.append(
                ChatMessage(
                    content=INTEGRATION_GUIDANCE_PROMPT.format(
                        planner_json=self.planner_json_str or "{}",
                        lib_consultant_json=self.lib_consult_json_str or "{}",
                    ),
                    role=MessageRole.USER,
                )
            )

        if (
            isinstance(self.token_counter, TokenCounterCached)
            and self.token_counter.enable_cache
        ):
            self.token_counter.add_cache_tag(ret[-1])
        return ret

    def get_order_prompt_messages(self) -> List[ChatMessage]:
        return [
            ChatMessage(
                content=ORDER_PROMPT.format(
                    output_format="".join(json.dumps(EXAMPLE_OUTPUT, indent=4))
                )
                + EXTRA_ORDER_PROMPT,
                role=MessageRole.USER,
            ),
        ]

    def get_format_error_prompt_messages(
        self, format_error: str, rtl_code: str
    ) -> List[ChatMessage]:
        return [
            ChatMessage(
                content=FORMAT_ERROR_PROMPT.format(
                    format_error=format_error, module_with_lineno=add_lineno(rtl_code)
                ),
                role=MessageRole.USER,
            ),
        ]

    def parse_output(
        self, response: Union[ChatResponse, AgentChatResponse]
    ) -> RTLOutputFormat:
        try:
            if isinstance(response, ChatResponse):
                raw_text = response.message.content
            elif isinstance(response, AgentChatResponse):
                raw_text = response.response
            else:
                raise TypeError(f"Unexpected response type: {type(response)}")
            output_json_obj: Dict = json.loads(raw_text, strict=False)
            ret = RTLOutputFormat(
                reasoning=output_json_obj["reasoning"], module=output_json_obj["module"]
            )
        except json.decoder.JSONDecodeError as e:
            ret = RTLOutputFormat(reasoning=f"Json Decode Error: {str(e)}", module="")
        return ret

    def chat(
        self,
        input_spec: str,
        testbench: str,
        interface: str,
        rtl_path: str,
        enable_cache: bool = False,
    ) -> Tuple[bool, str, List[str]]:
        if isinstance(self.token_counter, TokenCounterCached):
            self.token_counter.set_enable_cache(enable_cache)
        self.history = []
        self.token_counter.set_cur_tag(self.__class__.__name__)
        self.generated_tb = testbench
        self.generated_if = interface
        self.history.extend(self.get_init_prompt_messages(input_spec))

        for _ in range(self.max_trials):
            response = self.generate(self.history + self.get_order_prompt_messages())
            if isinstance(response, ChatResponse):
                message = response.message
            elif isinstance(response, AgentChatResponse):
                message = prompt_to_messages(response.response)
            else:
                raise TypeError(f"Unexpected response type: {type(response)}")

            resp_obj = self.parse_output(response)
            if resp_obj.reasoning.startswith("Json Decode Error"):
                logger.info(
                    f"RTL generation Error: {resp_obj.reasoning}, drop this response"
                )
                continue
            rtl_code = resp_obj.module
            rtl_lib_obj = _safe_json_loads(self.lib_consult_json_str)
            rtl_lib = _collect_rtl_libs(_append_rtl_libs(rtl_lib_obj))
            rtl_path_lib = os.path.join(os.path.dirname(rtl_path), "rtl_lib.sv")
            with open(rtl_path, "w") as f:
                f.write(rtl_code)
            with open(rtl_path_lib, "w") as f:
                f.write(rtl_lib)
            # We want to compile/elaborate the generated RTL and the library modules
            slang_ok, slang_out = compile_slang(
                rtl_path=rtl_path, mode="elab", top="TopModule"
            )
            iver_ok, iver_out = check_syntax(rtl_path=rtl_path, simulator="questa")

            # Concatenate logs so the LLM can address all errors in one go
            syntax_output = (
                "==== slang (syntax/elaboration) ====\n"
                f"{slang_out}\n\n"
                "==== iverilog (syntax) ====\n"
                f"{iver_out}\n"
            )

            # Treat as pass only if BOTH tools pass (so we fix anything either
            # tool flags)
            syntax_correct = slang_ok and iver_ok

            if syntax_correct:
                break

            # Feed both tools' messages back to the LLM for a combined fix attempt
            self.history.extend(
                [message]
                + self.get_format_error_prompt_messages(syntax_output, rtl_code)
            )
        return (syntax_correct, rtl_code, rtl_lib)

    def gen_candidates(
        self,
        input_spec: str,
        testbench: str,
        interface: str,
        rtl_path: str,
        candidates_num: int,
        enable_cache: bool = False,
    ) -> List[Tuple[bool, str]]:
        if isinstance(self.token_counter, TokenCounterCached):
            self.token_counter.set_enable_cache(enable_cache)
        self.history = []
        self.token_counter.set_cur_tag(self.__class__.__name__)
        self.generated_tb = testbench
        self.generated_if = interface
        self.history.extend(self.get_init_prompt_messages(input_spec))
        ret: List[Tuple[bool, str]] = [(False, "") for _ in range(candidates_num)]
        messages = [
            self.history + self.get_order_prompt_messages()
            for _ in range(candidates_num)
        ]
        logger.info(f"gen_candidates init input message: {messages[0]}")
        init_responses = self.batch_generate(messages)
        for i, response in enumerate(init_responses):
            rtl_code = self.parse_output(response).module
            if isinstance(response, ChatResponse):
                message = response.message
            elif isinstance(response, AgentChatResponse):
                message = prompt_to_messages(response.response)

            candidate_history: List[ChatMessage] = [message]
            for j in range(self.max_trials):
                with open(rtl_path, "w") as f:
                    f.write(rtl_code)

                slang_ok, slang_out = compile_slang(rtl_path=rtl_path, mode="elab")
                iver_ok, iver_out = check_syntax(rtl_path=rtl_path, simulator="questa")

                syntax_correct = slang_ok and iver_ok
                syntax_output = (
                    "==== slang (syntax/elaboration) ====\n"
                    f"{slang_out}\n\n"
                    "==== iverilog (syntax) ====\n"
                    f"{iver_out}\n"
                )

                ret[i] = (syntax_correct, rtl_code)
                logger.info(
                    f"Candidate {i + 1} / {candidates_num} trial {j + 1} / {self.max_trials} "
                    f"syntax_correct: {syntax_correct}"
                )
                logger.info(f"RTL code: {rtl_code}")

                if syntax_correct:
                    break

                if j < self.max_trials - 1:
                    candidate_history.extend(
                        self.get_format_error_prompt_messages(syntax_output, rtl_code)
                    )
                    response = self.generate(
                        self.history
                        + candidate_history
                        + self.get_order_prompt_messages()
                    )
                    rtl_code = self.parse_output(response).module
        return ret

    def ablation_chat(self, input_spec: str, rtl_path: str) -> Tuple[bool, str]:
        if isinstance(self.token_counter, TokenCounterCached):
            self.token_counter.set_enable_cache(False)
        self.history = []
        self.token_counter.set_cur_tag(self.__class__.__name__)
        self.generated_tb = None
        self.generated_if = None
        self.history.extend(self.get_init_prompt_messages(input_spec))
        syntax_correct = False
        rtl_code = ""
        for _ in range(self.max_trials):
            # Don't add order message into history, to save token
            response = self.generate(self.history + self.get_order_prompt_messages())
            if isinstance(response, ChatResponse):
                message = response.message
            elif isinstance(response, AgentChatResponse):
                message = prompt_to_messages(response.response)
            else:
                raise TypeError(f"Unexpected response type: {type(response)}")
            self.history.append(message)
            rtl_code = self.parse_output(response).module

            with open(rtl_path, "w") as f:
                f.write(rtl_code)

            slang_ok, slang_out = compile_slang(rtl_path=rtl_path, mode="elab")
            iver_ok, iver_out = check_syntax(rtl_path=rtl_path, simulator="questa")

            syntax_correct = slang_ok and iver_ok
            if syntax_correct:
                break

            syntax_output = (
                "==== slang (syntax/elaboration) ====\n"
                f"{slang_out}\n\n"
                "==== iverilog (syntax) ====\n"
                f"{iver_out}\n"
            )
            self.history.extend(
                self.get_format_error_prompt_messages(syntax_output, rtl_code)
            )

        return (syntax_correct, rtl_code)
