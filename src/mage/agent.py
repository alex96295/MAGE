import json
import os
import re
import sys
import traceback
from typing import List, Tuple

from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.llms import LLM

from .design_planner import DesignPlanner
from .doc_utils import CorpusPaths, DocumentManager, ParserConfig
from .lib_consultant import LibConsultant
from .lint_reviewer import LintReviewer
from .log_utils import get_logger, set_log_dir, switch_log_to_file, switch_log_to_stdout
from .rtl_editor import RTLEditor
from .rtl_generator import RTLGenerator
from .sim_judge import SimJudge
from .sim_reviewer import SimReviewer
from .style_reviewer import StyleReviewer
from .tb_generator import TBGenerator
from .token_counter import TokenCounter, TokenCounterCached

logger = get_logger(__name__)


class TopAgent:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.embed_model = resolve_embed_model("local:BAAI/bge-m3")
        self.token_counter = (
            TokenCounterCached(llm)
            if TokenCounterCached.is_cache_enabled(llm)
            else TokenCounter(llm)
        )
        self.sim_max_retry = 4
        self.rtl_max_candidates = 20
        self.rtl_selected_candidates = 2
        self.is_ablation = False
        self.redirect_log = False
        self.output_path = "./output"
        self.log_path = "./log"
        self.assets_path = "./assets"
        self.lib_path = "./pulp-verilog-eval/out/lib"
        self.assets_docs_path = "./.documents"
        self.assets_docs_dict = {}
        self.golden_tb_path: str | None = None
        self.golden_rtl_blackbox_path: str | None = None
        self.tb_gen: TBGenerator | None = None
        self.rtl_gen: RTLGenerator | None = None
        self.sim_reviewer: SimReviewer | None = None
        self.sim_judge: SimJudge | None = None
        self.rtl_edit: RTLEditor | None = None
        self.style_reviewer: StyleReviewer | None = None
        self.lint_reviewer: LintReviewer | None = None

        self.design_planner: DesignPlanner | None = None
        self.lib_consultant: LibConsultant | None = None

        self._design_plan_text: str | None = None
        self._lib_hints_text: str | None = None

    def prepare_documents(self) -> None:
        """Parse or load documents from assets/ subdirectories and store them as JSON."""
        os.makedirs(self.assets_docs_path, exist_ok=True)

        for assets_subdir in os.listdir(self.assets_path):
            assets_subdir_path = os.path.join(self.assets_path, assets_subdir)
            if not os.path.isdir(assets_subdir_path):
                continue  # skip files, only handle subdirectories

            # json filename for this subdir
            assets_subdir_json_path = os.path.join(
                self.assets_docs_path, f"{assets_subdir}.json"
            )

            docs_mgr = DocumentManager(
                paths=CorpusPaths(docs_dir=assets_subdir_path),
                parser_cfg=ParserConfig(parser_type="docling", use_gpu=False),
                docs_exts=[".pdf", ".pptx", ".ppt", ".md", ".docx", ".html"],
            )

            if not os.path.exists(assets_subdir_json_path):
                logger.info(f"TopAgent Parsing documents from {assets_subdir_path}")
                docs = docs_mgr.parse_all_files_in(assets_subdir_path)
                docs_mgr.save_documents(docs, assets_subdir_json_path)
            else:
                logger.info(
                    f"TopAgent Loading cached documents from {assets_subdir_json_path}"
                )
                docs = docs_mgr.load_documents(assets_subdir_json_path)

            # store in dict
            self.assets_docs_dict[assets_subdir] = docs or []

            # log summary
            logger.info(
                f"TopAgent Prepared {len(self.assets_docs_dict[assets_subdir])} documents "
                f"for assets subdirectory '{assets_subdir}'"
            )

    def set_assets_path(self, assets_path: str) -> None:
        self.assets_path = assets_path

    def set_lib_path(self, lib_path: str) -> None:
        self.lib_path = lib_path

    def set_output_path(self, output_path: str) -> None:
        self.output_path = output_path

    def set_log_path(self, log_path: str) -> None:
        self.log_path = log_path

    def set_ablation(self, is_ablation: bool) -> None:
        self.is_ablation = is_ablation

    def set_redirect_log(self, new_value: bool) -> None:
        self.redirect_log = new_value
        if self.redirect_log:
            switch_log_to_file()
        else:
            switch_log_to_stdout()

    def write_output(self, content: str, file_name: str) -> None:
        assert self.output_dir_per_run
        with open(f"{self.output_dir_per_run}/{file_name}", "w") as f:
            f.write(content)

    def _run_design_planning_phase(self, spec: str) -> Tuple[str, str]:
        """
        Returns (design_plan_text, lib_hints_text).
        These strings are persisted to disk and cached on the instance for later consumption.
        """
        assert self.design_planner is not None
        assert self.lib_consultant is not None

        # Plan the design
        logger.info("[DesignPlanner] Starting design planning phase.")
        design_plan_json = self.design_planner.chat(spec)
        logger.info("[DesignPlanner] Plan produced and saved.")

        # Library/IP recommendations
        logger.info("[LibConsultant] Starting library consulting phase.")
        lib_hints_json = self.lib_consultant.consult(
            design_plan=json.loads(design_plan_json)
        )
        logger.info("[LibConsultant] Library notes produced.")

        # Cache and persist
        self._design_plan_json = design_plan_json or ""
        self._lib_hints_json = lib_hints_json or ""

        try:
            self.write_output(self._design_plan_json, "design_plan.json")
            self.write_output(self._lib_hints_json, "lib_hints.json")
        except Exception:
            logger.warning("Failed to persist planning artifacts.", exc_info=True)

        return self._design_plan_json, self._lib_hints_json

    def run_instance(self, spec: str) -> Tuple[bool, str]:
        """
        Run a single instance of the benchmark
        Return value:
        - is_pass: bool, whether the instance passes the golden testbench
        - rtl_code: str, the generated RTL code
        """
        assert self.tb_gen
        assert self.rtl_gen
        assert self.sim_reviewer
        assert self.sim_judge
        assert self.rtl_edit

        plan_text, lib_text = self._run_design_planning_phase(spec)

        self.tb_gen.reset()
        self.tb_gen.set_golden_tb_path(self.golden_tb_path)
        if not self.golden_tb_path:
            logger.info("No golden testbench provided")
        testbench, interface = self.tb_gen.chat(spec)
        logger.info("Initial tb:")
        logger.info(testbench)
        logger.info("Initial if:")
        logger.info(interface)
        self.write_output(testbench, "tb.sv")
        self.write_output(interface, "if.sv")
        self.rtl_gen.reset()
        logger.info(spec)

        # Pass plan/lib hints
        is_syntax_pass, rtl_code, _ = self.rtl_gen.chat(
            input_spec=spec,
            testbench=testbench,
            interface=interface,
            rtl_path=os.path.join(self.output_dir_per_run, "rtl.sv"),
            design_plan=plan_text,
            lib_hints=lib_text,
        )
        if not is_syntax_pass:
            return False, rtl_code
        self.write_output(rtl_code, "rtl.sv")
        logger.info("Initial rtl:")
        logger.info(rtl_code)

        tb_need_fix = True
        rtl_need_fix = True
        sim_log = ""
        for i in range(self.sim_max_retry):
            # run simulation judge, overwrite is_sim_pass
            is_sim_pass, sim_mismatch_cnt, sim_log = self.sim_reviewer.review(
                simulator="questa"
            )
            if is_sim_pass:
                tb_need_fix = False
                rtl_need_fix = False
                break
            self.sim_judge.reset()
            tb_need_fix = self.sim_judge.chat(spec, sim_log, rtl_code, testbench)
            if tb_need_fix:
                self.tb_gen.reset()
                if i == 0:
                    self.tb_gen.gen_display_queue = False
                    logger.info("Fallback from display queue to display moment")
                else:
                    self.tb_gen.set_failed_trial(sim_log, rtl_code, testbench)

                testbench, _ = self.tb_gen.chat(spec)
                self.write_output(testbench, "tb.sv")
                logger.info("Revised tb:")
                logger.info(testbench)
            else:
                break

        assert not tb_need_fix, f"tb_need_fix should be False. sim_log: {sim_log}"

        candidates_info: List[Tuple[str, int, str]] = []
        if rtl_need_fix:
            # Candidates Generation
            assert (
                sim_mismatch_cnt > 0
            ), f"rtl_need_fix should be True only when sim_mismatch_cnt > 0. sim_log: {sim_log}"
            self.rtl_gen.reset()

            # seed with cache-enabled attempt including plan/lib hints
            candidates = [
                self.rtl_gen.chat(
                    input_spec=spec,
                    testbench=testbench,
                    interface=interface,
                    rtl_path=os.path.join(self.output_dir_per_run, "rtl.sv"),
                    enable_cache=True,
                    design_plan=self._design_plan_text,
                    lib_hints=self._lib_hints_text,
                )
            ]  # Write Cache
            if self.rtl_max_candidates > 1:
                candidates += self.rtl_gen.gen_candidates(
                    input_spec=spec,
                    testbench=testbench,
                    interface=interface,
                    rtl_path=os.path.join(self.output_dir_per_run, "rtl.sv"),
                    candidates_num=self.rtl_max_candidates - 1,
                    enable_cache=True,
                )
            for i in range(self.rtl_max_candidates):
                logger.info(
                    f"Candidate generation: round {i + 1} / {self.rtl_max_candidates}"
                )
                is_syntax_pass_candiate, rtl_code_candidate = candidates[i]
                if not is_syntax_pass_candiate:
                    continue
                self.write_output(rtl_code_candidate, "rtl.sv")
                is_sim_pass_candidate, sim_mismatch_cnt_candidate, sim_log_candidate = (
                    self.sim_reviewer.review(simulator="questa")
                )
                if is_sim_pass_candidate:
                    rtl_code = rtl_code_candidate
                    sim_mismatch_cnt = sim_mismatch_cnt_candidate
                    sim_log = sim_log_candidate
                    rtl_need_fix = False
                    break
                candidates_info.append(
                    (rtl_code_candidate, sim_mismatch_cnt_candidate, sim_log_candidate)
                )

        candidates_info.sort(key=lambda x: x[1])
        candidates_info_unique_sign = set()
        candidates_info_unique = []
        for candidate in candidates_info:
            if candidate[1] not in candidates_info_unique_sign:
                candidates_info_unique_sign.add(candidate[1])
                candidates_info_unique.append(candidate)

        if rtl_need_fix:
            # Editor iteration
            for i in range(self.rtl_selected_candidates):
                logger.info(
                    f"Selected candidate: round {i + 1} / {self.rtl_selected_candidates}"
                )
                i = i % len(candidates_info_unique)
                rtl_code, sim_mismatch_cnt, sim_log = candidates_info_unique[i]
                with open(f"{self.output_dir_per_run}/rtl.sv", "w") as f:
                    f.write(rtl_code)
                self.rtl_edit.reset()
                is_sim_pass, rtl_code = self.rtl_edit.chat(
                    spec=spec,
                    output_dir_per_run=self.output_dir_per_run,
                    sim_failed_log=sim_log,
                    sim_mismatch_cnt=sim_mismatch_cnt,
                )
                if is_sim_pass:
                    rtl_need_fix = False
                    break

        if not is_sim_pass:  # Run if keep failing before last try
            is_sim_pass, _, _ = self.sim_reviewer.review(simulator="questa")

        # policy/consistency check vs style guide (based on descriptive rules)
        if is_sim_pass and self.style_reviewer is not None:
            logger.info(
                "[StyleReviewer] Starting style-only pass on working design and TB."
            )
            original_rtl = rtl_code
            original_tb = testbench

            styled_rtl, styled_tb, style_summary, style_reason = (
                self.style_reviewer.review(
                    rtl_code=rtl_code,
                    tb_code=testbench,
                    enable_cache=True,
                )
            )

            # Write styled sources
            self.write_output(styled_rtl, "rtl.sv")
            self.write_output(styled_tb, "tb.sv")

            # Re-run sim to ensure no behavior change
            style_sim_pass, _, style_sim_log = self.sim_reviewer.review(
                simulator="questa"
            )
            if style_sim_pass:
                logger.info("[StyleReviewer] Simulation PASSED after styling.")
                rtl_code = styled_rtl
                testbench = styled_tb
                # Optionally persist a summary
                try:
                    with open(
                        f"{self.output_dir_per_run}/style_changes.json", "w"
                    ) as f:
                        json.dump(
                            {
                                "reasoning": style_reason,
                                "change_summary": style_summary,
                            },
                            f,
                            indent=2,
                        )
                except Exception:
                    pass
            else:
                logger.warning(
                    "[StyleReviewer] Simulation FAILED after styling. Reverting to originals."
                )
                # Revert files to originals
                self.write_output(original_rtl, "rtl.sv")
                self.write_output(original_tb, "tb.sv")
                # Ensure sim still passes with originals (it should)
                _ = self.sim_reviewer.review(simulator="questa")

        # rule-driven lint/formatting
        if is_sim_pass and self.lint_reviewer is not None:
            logger.info(
                "[LintReviewer] Starting format+lint pass on working design and TB."
            )
            # Read current (potentially styled) files as input to lint reviewer
            with open(f"{self.output_dir_per_run}/rtl.sv", "r") as f:
                lr_in_rtl = f.read()
            with open(f"{self.output_dir_per_run}/tb.sv", "r") as f:
                lr_in_tb = f.read()

            (
                final_rtl,
                final_tb,
                lint_ok,
                lint_log,
                fmt_log,
                lint_changes,
                lint_reason,
            ) = self.lint_reviewer.review(
                rtl_code=lr_in_rtl,
                tb_code=lr_in_tb,
                output_dir_per_run=self.output_dir_per_run,
                rtl_filename="rtl.sv",
                tb_filename="tb.sv",
                enable_llm_fix=True,
            )

            # Re-run sim to ensure no behavior change post-lint/format
            lint_sim_pass, _, lint_sim_log = self.sim_reviewer.review(
                simulator="questa"
            )
            if lint_sim_pass:
                logger.info("[LintReviewer] Simulation PASSED after format+lint.")
                rtl_code = final_rtl
                testbench = final_tb
                # Save logs/summaries
                try:
                    with open(f"{self.output_dir_per_run}/lint_changes.json", "w") as f:
                        json.dump(
                            {
                                "reasoning": lint_reason,
                                "change_summary": lint_changes,
                                "lint_passed": bool(lint_ok),
                                "lint_log": (
                                    json.loads(lint_log)
                                    if isinstance(lint_log, str)
                                    else str(lint_log)
                                ),
                                "format_log": (
                                    json.loads(fmt_log)
                                    if isinstance(fmt_log, str)
                                    else str(fmt_log)
                                ),
                            },
                            f,
                            indent=2,
                        )
                except Exception:
                    pass
            else:
                logger.warning(
                    "[LintReviewer] Simulation FAILED after format+lint. Reverting to pre-lint sources."
                )
                # Revert to pre-lint (the versions already on disk before LintReviewer started)
                self.write_output(lr_in_rtl, "rtl.sv")
                self.write_output(lr_in_tb, "tb.sv")
                # Sanity: re-run sim to confirm ok
                _ = self.sim_reviewer.review(simulator="questa")

        return is_sim_pass, rtl_code

    def run_instance_ablation(self, spec: str) -> Tuple[bool, str]:
        """
        Run a single instance of the benchmark in ablation mode
        Return value:
        - is_pass: bool, whether the instance passes the golden testbench
        - rtl_code: str, the generated RTL code
        """
        assert self.rtl_gen

        if self.design_planner and self.lib_consultant:
            self._run_design_planning_phase(spec)

        self.rtl_gen.reset()
        logger.info(spec)
        # Current ablation: only run RTL generation with syntax check
        is_syntax_pass, rtl_code = self.rtl_gen.ablation_chat(
            input_spec=spec, rtl_path=os.path.join(self.output_dir_per_run, "rtl.sv")
        )
        self.write_output(rtl_code, "rtl.sv")
        return is_syntax_pass, rtl_code

    def _run(self, spec: str) -> Tuple[bool, str]:
        try:
            if os.path.exists(f"{self.output_dir_per_run}/properly_finished.tag"):
                os.remove(f"{self.output_dir_per_run}/properly_finished.tag")

            # prepare documents for rag for all the agents
            self.prepare_documents()
            logger.info(self.assets_docs_dict.keys())
            for subdir, docs in self.assets_docs_dict.items():
                logger.info(f"Subdir: {subdir}, Docs: {len(docs)}")

            # initialize all the agents
            self.token_counter.reset()
            self.sim_reviewer = SimReviewer(
                self.output_dir_per_run,
                self.golden_rtl_blackbox_path,
            )
            self.rtl_gen = RTLGenerator(self.token_counter)
            self.tb_gen = TBGenerator(self.token_counter)
            self.sim_judge = SimJudge(self.token_counter)
            self.rtl_edit = RTLEditor(
                self.token_counter, sim_reviewer=self.sim_reviewer
            )
            self.style_reviewer = StyleReviewer(self.token_counter)
            self.lint_reviewer = LintReviewer()

            self.design_planner = DesignPlanner(self.token_counter)
            self.lib_consultant = LibConsultant(self.embed_model)

            # tune docs for rag based on the agent. Key idea is that not all
            # the agents need the same knowledge (divide et impera)
            language_docs = self.assets_docs_dict.get("language", [])
            style_docs = self.assets_docs_dict.get("style_guide", [])

            tb_gen_docs = list(language_docs)
            rtl_gen_docs = list(language_docs)
            style_reviewer_docs = list(style_docs)

            # init rag for agents
            memory_padding = 256

            logger.info("Initialize RAG for agents")
            self.tb_gen.init_rag(
                persist_dir="./.vector_storage/tb_gen",
                faiss_path="./.faiss_storage/tb_gen_faiss.bin",
                docs=tb_gen_docs,
                embed_model=self.embed_model,
                memory_token_limit=self.llm.metadata.context_window - memory_padding,
            )

            self.rtl_gen.init_rag(
                persist_dir="./.vector_storage/rtl_gen",
                faiss_path="./.faiss_storage/rtl_gen_faiss.bin",
                docs=rtl_gen_docs,
                embed_model=self.embed_model,
            )

            self.rtl_edit.init_rag(
                persist_dir="./.vector_storage/rtl_gen",
                faiss_path="./.faiss_storage/rtl_gen_faiss.bin",
                docs=rtl_gen_docs,
                embed_model=self.embed_model,
            )

            self.style_reviewer.init_rag(
                persist_dir="./.vector_storage/style_reviewer",
                faiss_path="./.faiss_storage/style_reviewer_faiss.bin",
                docs=style_reviewer_docs,
                embed_model=self.embed_model,
                memory_token_limit=self.llm.metadata.context_window - memory_padding,
            )

            self.design_planner.init_rag(
                persist_dir="./.vector_storage/rtl_gen",
                faiss_path="./.faiss_storage/rtl_gen_faiss.bin",
                docs=rtl_gen_docs,
                embed_model=self.embed_model,
                memory_token_limit=self.llm.metadata.context_window - memory_padding,
            )

            self.lib_consultant.ingest_from_dir(self.lib_path)
            self.lib_consultant.build_index()

            # configure lint reviewer
            self.lint_reviewer.set_lint_autofix(mode="inplace")

            ret = (
                self.run_instance(spec)
                if not self.is_ablation
                else self.run_instance_ablation(spec)
            )
            self.token_counter.log_token_stats()
            with open(f"{self.output_dir_per_run}/properly_finished.tag", "w") as f:
                f.write("1")
        except Exception:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            ret = False, f"Exception: {exc_info[1]}"
        return ret

    def run(
        self,
        benchmark_type_name: str,
        task_id: str,
        spec: str,
        golden_tb_path: str | None = None,
        golden_rtl_blackbox_path: str | None = None,
    ) -> Tuple[bool, str]:
        self.golden_tb_path = golden_tb_path
        self.golden_rtl_blackbox_path = golden_rtl_blackbox_path
        log_dir_per_run = f"{self.log_path}/{benchmark_type_name}_{task_id}"
        self.output_dir_per_run = f"{self.output_path}/{benchmark_type_name}_{task_id}"
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.output_dir_per_run, exist_ok=True)
        set_log_dir(log_dir_per_run)
        if self.redirect_log:
            with open(f"{log_dir_per_run}/mage_rtl.log", "w") as f:
                sys.stdout = f
                sys.stderr = f
                result = self._run(spec)
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        else:
            result = self._run(spec)
        # Redirect log contains format with rich text.
        # Provide a rich-free version for log parsing or less viewing.
        if self.redirect_log:
            with open(f"{log_dir_per_run}/mage_rtl.log", "r") as f:
                content = f.read()
            content = re.sub(r"\[.*?m", "", content)
            with open(f"{log_dir_per_run}/mage_rtl_rich_free.log", "w") as f:
                f.write(content)
        return result
