import json
import re
from typing import List, Optional

import anthropic
from llama_index.llms.anthropic import Anthropic


def add_lineno(file_content: str) -> str:
    lines = file_content.split("\n")
    ret = ""
    for i, line in enumerate(lines):
        ret += f"{i+1}: {line}\n"
    return ret


def reformat_json_string(output: str) -> str:
    # in gemini, the output has markdown surrounding the json string
    # like ```json ... ```
    # we need to remove the markdown
    # remove by using regex between ```json and ```
    pattern = r"```json(.*?)```"
    match = re.search(pattern, output, re.DOTALL)
    if match:
        return match.group(1).strip()

    pattern = r"```xml(.*?)```"
    match = re.search(pattern, output, re.DOTALL)
    if match:
        return match.group(1).strip()

    return output.strip()


def _safe_json_loads(s: Optional[str]) -> dict:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def _collect_rtl_libs(lib_consult_obj: dict) -> List[str]:
    """
    Collect SystemVerilog snippets for reused modules from the lib consultant output.

    Expected flexible shapes (we tolerate several):
    - out['reuse'][*]['library_json']['content']['reuse'] -> str (entire module or stub)
    - out['reuse'][*]['library_json']['content']['declaration'] -> str
    - out['reuse'][*]['library_json']['sv_declaration'] -> str
    - out['reuse'][*]['library_json']['sv'] -> str
    - out['reuse'][*]['library_json']['body'] -> str
    If none present, return [].
    """
    decls: List[str] = []
    reuse_bucket = (lib_consult_obj or {}).get("reuse", {}) or {}
    for _, entry in reuse_bucket.items():
        libj = (entry or {}).get("library_json")
        if not isinstance(libj, dict):
            continue
        content = libj.get("content", {}) or {}
        candidates = [
            content.get("reuse"),
            content.get("declaration"),
            libj.get("sv_declaration"),
            libj.get("sv"),
            libj.get("body"),
        ]
        for c in candidates:
            if isinstance(c, str) and c.strip():
                decls.append(c.strip())
                break
    # De-dup by module name if possible
    uniq: List[str] = []
    seen_mods: set[str] = set()
    modname_re = re.compile(r"^\s*module\s+([A-Za-z_]\w*)", re.M)
    for d in decls:
        m = modname_re.search(d)
        key = m.group(1) if m else d[:80]
        if key not in seen_mods:
            seen_mods.add(key)
            uniq.append(d)
    return uniq


def _append_rtl_libs(reuse_decls: List[str]) -> str:
    return ("\n\n".join(reuse_decls)) + "\n"


class VertexAnthropicWithCredentials(Anthropic):
    def __init__(self, credentials, **kwargs):
        """
        In addition to all parameters accepted by Anthropic, this class accepts a
        new parameter `credentials` that will be passed to the underlying clients.
        """
        # Pop parameters that determine client type so we can reuse them in our branch.
        region = kwargs.get("region")
        project_id = kwargs.get("project_id")
        aws_region = kwargs.get("aws_region")

        # Call the parent initializer; this sets up a default _client and _aclient.
        super().__init__(**kwargs)

        # If using AnthropicVertex (i.e., region and project_id are provided and aws_region is None),
        # override the _client and _aclient with the additional credentials parameter.
        if region and project_id and not aws_region:
            self._client = anthropic.AnthropicVertex(
                region=region,
                project_id=project_id,
                credentials=credentials,  # extra argument
                timeout=self.timeout,
                max_retries=self.max_retries,
                default_headers=kwargs.get("default_headers"),
            )
            self._aclient = anthropic.AsyncAnthropicVertex(
                region=region,
                project_id=project_id,
                credentials=credentials,  # extra argument
                timeout=self.timeout,
                max_retries=self.max_retries,
                default_headers=kwargs.get("default_headers"),
            )
        # Optionally, you could add similar overrides for the aws_region branch if needed.
