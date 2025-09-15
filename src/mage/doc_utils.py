# doc_utils.py
# Utilities for parsing, saving, loading, and cleaning document corpora with LlamaIndex.

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

# --- Optional imports, handled gracefully ---
_missing: List[str] = []
try:
    from llama_index.core import Document, SimpleDirectoryReader
except Exception:
    _missing.append("llama_index")

    class Document:  # type: ignore
        def __init__(
            self,
            text: str,
            doc_id: Optional[str] = None,
            metadata: Optional[dict] = None,
        ):
            self.text = text
            self._doc_id = doc_id
            self.metadata = metadata or {}

        def get_doc_id(self):
            return self._doc_id


try:
    from llama_parse import LlamaParse

    _HAVE_LLAMA_PARSE = True
except Exception:
    _HAVE_LLAMA_PARSE = False

try:
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
    )
    from llama_index.readers.docling import DoclingReader

    _HAVE_DOCLING = True
except Exception:
    _HAVE_DOCLING = False

logger = logging.getLogger(__name__)


@dataclass
class ParserConfig:
    """
    Configuration for document parsing.
    parser_type: "llamaparse" or "docling".
    use_gpu: whether to allow GPU usage (only applies to docling).
    """

    parser_type: str = "docling"  # default to docling
    use_gpu: bool = True  # default is to use GPU if available


@dataclass
class CorpusPaths:
    """Directories & files used in your corpus lifecycle."""

    docs_dir: str = "./.documents"


@dataclass
class DocumentManager:
    """
    High-level manager for parsing corpora into LlamaIndex Documents,
    plus JSON save/load and folder clean-up utilities.
    """

    paths: CorpusPaths = field(default_factory=CorpusPaths)
    parser_cfg: ParserConfig = field(default_factory=ParserConfig)
    docs_exts: Sequence[str] = field(
        default_factory=lambda: [".pdf", ".pptx", ".ppt", ".md", ".docx", ".html"]
    )

    # Internal: resolved parser and file_extractor map
    _parser: object = field(init=False, default=None)
    _file_extractor: Dict[str, object] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._ensure_dirs()
        self._resolve_parser()
        self._build_file_extractor()

    # -------------------- Public API --------------------

    @property
    def file_extractor(self) -> Dict[str, object]:
        """Mapping of file extensions to the active parser/reader."""
        return self._file_extractor

    def parse_all_files_in(
        self, folder_path: str, num_workers: int = 32
    ) -> List[Document]:
        """
        Parse all files in `folder_path` using SimpleDirectoryReader + chosen parser.
        Only files with extensions defined in `docs_exts` are kept.
        """
        self._require(
            ["llama_index"], message="Parsing requires llama_index installed."
        )
        valid_exts = set(self._file_extractor.keys())

        # Remove files not matching valid extensions
        for root, _, files in os.walk(folder_path):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in valid_exts:
                    full_path = os.path.join(root, fname)
                    logger.info(f"Removing file not in valid extensions: {full_path}")
                    try:
                        os.remove(full_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove {full_path}: {e}")

        logger.info(f"Parsing all files found in {folder_path} ...")
        reader = SimpleDirectoryReader(
            input_dir=folder_path,
            file_extractor=self._file_extractor,
            exclude_hidden=False,
        )
        docs: List[Document] = reader.load_data(
            show_progress=True, num_workers=num_workers
        )  # type: ignore
        return docs

    def save_documents(self, documents: Sequence[Document], filename: str) -> None:
        """Serialize Documents to JSON (doc_id, text, metadata)."""
        payload = []
        for doc in documents:
            doc_id = getattr(doc, "doc_id", None)
            if doc_id is None and hasattr(doc, "get_doc_id"):
                doc_id = doc.get_doc_id()
            payload.append(
                {
                    "doc_id": doc_id,
                    "text": getattr(doc, "text", ""),
                    "metadata": getattr(doc, "metadata", {}) or {},
                }
            )
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(payload)} documents to {filename}.")

    def load_documents(self, filename: str) -> Optional[List[Document]]:
        """Load Documents from JSON created by `save_documents`."""
        if not os.path.exists(filename):
            logger.warning(f"No document file found at {filename}.")
            return None
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs: List[Document] = []
        for item in data:
            text = item.get("text", "")
            doc_id = item.get("doc_id")
            metadata = item.get("metadata", {}) or {}
            docs.append(
                Document(text=text, doc_id=doc_id, metadata=metadata)
            )  # type: ignore
        logger.info(f"Loaded {len(docs)} documents from {filename}.")
        return docs

    def cleanup_folder(self, folder_path: str, remove_dir: bool = False) -> None:
        """Remove files in a folder (and optionally the folder itself)."""
        try:
            if not os.path.exists(folder_path):
                return
            if remove_dir:
                shutil.rmtree(folder_path, ignore_errors=True)
                logger.info(f"Removed folder: {folder_path}")
                return
            for root, _, files in os.walk(folder_path):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    try:
                        os.remove(fpath)
                    except Exception as e:
                        logger.warning(f"Could not remove {fpath}: {e}")
            logger.info(f"Cleaned up folder: {folder_path}")
        except Exception as e:
            logger.warning(f"Could not clean up {folder_path}: {e}")

    # -------------------- Internal helpers --------------------

    def _ensure_dirs(self) -> None:
        os.makedirs(self.paths.docs_dir, exist_ok=True)

    def _resolve_parser(self) -> None:
        """Decide which parser to use based on parser_cfg.parser_type."""
        if self.parser_cfg.parser_type == "llamaparse":
            if not _HAVE_LLAMA_PARSE:
                raise RuntimeError("llama_parse not installed. pip install llama-parse")
            if not os.getenv("LLAMA_CLOUD_API_KEY"):
                raise RuntimeError("LLAMA_CLOUD_API_KEY not set for LlamaParse")
            self._parser = LlamaParse()
            logger.info("Using LlamaParse as parser.")
        elif self.parser_cfg.parser_type == "docling":
            if not _HAVE_DOCLING:
                raise RuntimeError(
                    "docling reader not installed. pip install llama-index-readers-docling"
                )

            # Configure accelerator options depending on GPU usage flag
            if self.parser_cfg.use_gpu:
                accel_opts = AcceleratorOptions(device=AcceleratorDevice.GPU)
                logger.info("DoclingReader will use GPU.")
            else:
                accel_opts = AcceleratorOptions(device=AcceleratorDevice.CPU)
                logger.info("DoclingReader forced to use CPU only (no GPU).")

            pipeline_opts = PdfPipelineOptions(accelerator_options=accel_opts)

            self._parser = DoclingReader(
                export_type=DoclingReader.ExportType.JSON,
                pipeline_options=pipeline_opts,
            )
            logger.info("Using DoclingReader as parser.")
        else:
            raise ValueError(
                f"Unknown parser_type {self.parser_cfg.parser_type}. Must be 'llamaparse' or 'docling'."
            )

    def _build_file_extractor(self) -> None:
        """Build the file_extractor mapping from extensions to the chosen parser."""
        parser = self._parser
        if parser is None:
            raise RuntimeError("Parser not initialized.")
        self._file_extractor = {ext: parser for ext in self.docs_exts}

    def _require(self, pkgs: Iterable[str], message: str = "") -> None:
        missing = [p for p in pkgs if p == "llama_index" and "llama_index" in _missing]
        if missing:
            hint = "\nTry:\n  pip install llama-index-core llama-index-readers-docling llama-parse\n"
            raise RuntimeError(
                message or f"Missing packages: {', '.join(missing)}{hint}"
            )
