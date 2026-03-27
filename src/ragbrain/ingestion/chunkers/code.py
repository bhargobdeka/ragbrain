"""AST-aware code chunker using tree-sitter.

Parses source code into an Abstract Syntax Tree and extracts semantically
complete units — functions, classes, methods — never splitting mid-expression.

Architecture:
  ASTCodeChunker  → uses tree-sitter for Python/JS/TS/Go/Rust/Java
  _RegexFallback  → used when tree-sitter grammar is unavailable

References:
  - cAST paper (CMU 2025): https://arxiv.org/abs/2506.15655
  - tree-sitter: https://tree-sitter.github.io/tree-sitter/
  - code-chunk library (Supermemory): https://github.com/supermemoryai/code-chunk
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

try:
    from tree_sitter import Language, Node, Parser
    _TREE_SITTER_AVAILABLE = True
except ImportError:
    _TREE_SITTER_AVAILABLE = False

_MAX_CHUNK_CHARS = 3000

# Maps language name → tree-sitter grammar module name
_LANG_MODULE: dict[str, str] = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "go": "tree_sitter_go",
    "rust": "tree_sitter_rust",
    "java": "tree_sitter_java",
}

# AST node types that represent top-level semantic units per language
_CHUNK_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {
        "function_declaration", "class_declaration", "method_definition",
        "arrow_function", "export_statement",
    },
    "typescript": {
        "function_declaration", "class_declaration", "method_definition",
        "interface_declaration", "type_alias_declaration", "export_statement",
    },
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "enum_item", "trait_item"},
    "java": {"method_declaration", "class_declaration", "interface_declaration"},
}

# AST node types for import/use statements per language
_IMPORT_TYPES: dict[str, set[str]] = {
    "python": {"import_statement", "import_from_statement"},
    "javascript": {"import_statement", "import_declaration"},
    "typescript": {"import_statement", "import_declaration"},
    "go": {"import_declaration"},
    "rust": {"use_declaration"},
    "java": {"import_declaration"},
}


@dataclass
class CodeUnit:
    """A single extracted code unit (function, class, method, etc.)."""

    content: str
    name: str = ""
    scope_chain: str = ""      # e.g. "Trainer.train_step"
    docstring: str = ""        # first docstring/comment in the unit
    imports: list[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_text(node: "Node", src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _find_name(node: "Node", src: bytes) -> str:
    """Return the identifier name of a definition node."""
    for child in node.children:
        if child.type in ("identifier", "name", "type_identifier"):
            return _node_text(child, src)
    return ""


def _python_docstring(node: "Node", src: bytes) -> str:
    """Extract the leading docstring from a Python function/class body."""
    for child in node.children:
        if child.type == "block":
            for stmt in child.children:
                if stmt.type == "expression_statement":
                    for expr in stmt.children:
                        if expr.type == "string":
                            raw = _node_text(expr, src).strip()
                            # Strip triple/single quotes
                            for q in ('"""', "'''", '"', "'"):
                                if raw.startswith(q) and raw.endswith(q) and len(raw) > 2 * len(q):
                                    return raw[len(q):-len(q)].strip()
                            return raw
    return ""


def _collect_imports(root: "Node", src: bytes, lang: str) -> list[str]:
    """Gather all top-level import statements."""
    import_node_types = _IMPORT_TYPES.get(lang, set())
    return [
        _node_text(child, src).strip()
        for child in root.children
        if child.type in import_node_types
    ]


def _walk(
    node: "Node",
    src: bytes,
    lang: str,
    imports: list[str],
    parent_scope: str = "",
) -> list[CodeUnit]:
    """Recursively walk the AST and collect semantic units."""
    chunk_types = _CHUNK_TYPES.get(lang, set())
    units: list[CodeUnit] = []

    for child in node.children:
        # Unwrap decorated_definition in Python to get the real type
        target = child
        if child.type == "decorated_definition":
            for sub in child.children:
                if sub.type in ("function_definition", "class_definition"):
                    target = sub
                    break

        if child.type not in chunk_types and target.type not in chunk_types:
            continue

        name = _find_name(target, src)
        scope = f"{parent_scope}.{name}" if parent_scope and name else (name or parent_scope)

        docstring = ""
        if lang == "python" and target.type in ("function_definition", "class_definition"):
            docstring = _python_docstring(target, src)

        units.append(CodeUnit(
            content=_node_text(child, src),
            name=name,
            scope_chain=scope,
            docstring=docstring,
            imports=imports,
            start_line=child.start_point[0] + 1,
            end_line=child.end_point[0] + 1,
        ))

        # Recurse into class body nodes (block/class_body/declaration_list)
        # to also extract methods as separate sub-chunks.
        if target.type in ("class_definition", "class_declaration"):
            for class_child in target.children:
                if class_child.type in ("block", "class_body", "declaration_list"):
                    sub_units = _walk(class_child, src, lang, imports, parent_scope=scope)
                    units.extend(sub_units)

    return units


# ---------------------------------------------------------------------------
# Regex fallback (used when tree-sitter grammar not installed)
# ---------------------------------------------------------------------------

_BOUNDARY_PATTERNS: dict[str, re.Pattern] = {
    "python": re.compile(r"^(class |def |async def )", re.MULTILINE),
    "javascript": re.compile(r"^(function |const |class |async function )", re.MULTILINE),
    "typescript": re.compile(r"^(function |const |class |interface |type |async function )", re.MULTILINE),
    "java": re.compile(r"^(public |private |protected |class )", re.MULTILINE),
    "go": re.compile(r"^func ", re.MULTILINE),
    "rust": re.compile(r"^(fn |pub fn |impl |struct |enum )", re.MULTILINE),
}
_GENERIC_BOUNDARY = re.compile(r"\n{2,}")


class _RegexFallback:
    """Regex-based chunker — preserved as fallback for unsupported languages."""

    def chunk(self, code: str, language: str | None = None) -> list[str]:
        if not code.strip():
            return []
        pattern = _BOUNDARY_PATTERNS.get((language or "").lower(), _GENERIC_BOUNDARY)
        return self._split_on_pattern(code, pattern)

    def _split_on_pattern(self, code: str, pattern: re.Pattern) -> list[str]:
        matches = list(pattern.finditer(code))
        if not matches:
            return self.hard_split(code)
        chunks: list[str] = []
        boundaries = [m.start() for m in matches]
        if boundaries[0] > 0:
            header = code[: boundaries[0]].strip()
            if header:
                chunks.append(header)
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(code)
            segment = code[start:end].strip()
            if not segment:
                continue
            if len(segment) > _MAX_CHUNK_CHARS:
                chunks.extend(self.hard_split(segment))
            else:
                chunks.append(segment)
        return chunks or [code.strip()]

    def hard_split(self, code: str) -> list[str]:
        parts = re.split(r"\n{2,}", code)
        chunks: list[str] = []
        current = ""
        for part in parts:
            if len(current) + len(part) > _MAX_CHUNK_CHARS and current:
                chunks.append(current.strip())
                current = part
            else:
                current = (current + "\n\n" + part) if current else part
        if current.strip():
            chunks.append(current.strip())
        return chunks


# ---------------------------------------------------------------------------
# Public chunker
# ---------------------------------------------------------------------------

class ASTCodeChunker:
    """Parse code into semantic units using tree-sitter AST.

    Falls back to regex splitting if a language grammar is not installed.
    Both paths return ``list[CodeUnit]`` for a consistent interface.

    Usage::

        chunker = ASTCodeChunker()
        units = chunker.chunk(source_code, language="python")
        for unit in units:
            print(unit.scope_chain, unit.docstring)
    """

    def __init__(self) -> None:
        self._parsers: dict[str, "Parser"] = {}
        self._fallback = _RegexFallback()

    # Public alias so the router can keep ``CodeChunker`` references working
    CodeChunker = None  # set below

    def _get_parser(self, lang: str) -> "Parser | None":
        if not _TREE_SITTER_AVAILABLE:
            return None
        if lang not in self._parsers:
            module_name = _LANG_MODULE.get(lang)
            if not module_name:
                return None
            try:
                import importlib
                mod = importlib.import_module(module_name)
                self._parsers[lang] = Parser(Language(mod.language()))
            except Exception:
                return None
        return self._parsers[lang]

    def chunk(self, code: str, language: str | None = None) -> list[CodeUnit]:
        """Split code into semantic units.

        Args:
            code: Raw source code string.
            language: Language hint (``"python"``, ``"typescript"``, …).

        Returns:
            List of :class:`CodeUnit`.  Empty list if ``code`` is blank.
        """
        if not code.strip():
            return []

        lang = (language or "").lower()
        parser = self._get_parser(lang) if lang else None

        if parser is None:
            texts = self._fallback.chunk(code, language)
            return [CodeUnit(content=t) for t in texts]

        src_bytes = code.encode("utf-8")
        tree = parser.parse(src_bytes)
        imports = _collect_imports(tree.root_node, src_bytes, lang)
        units = _walk(tree.root_node, src_bytes, lang, imports)

        if not units:
            return [CodeUnit(content=code.strip(), imports=imports)]

        # Split oversized units without losing metadata
        result: list[CodeUnit] = []
        for unit in units:
            if len(unit.content) > _MAX_CHUNK_CHARS:
                sub_texts = self._fallback.hard_split(unit.content)
                for idx, sub in enumerate(sub_texts):
                    result.append(CodeUnit(
                        content=sub,
                        name=unit.name,
                        scope_chain=f"{unit.scope_chain}[part{idx}]" if unit.scope_chain else "",
                        docstring=unit.docstring if idx == 0 else "",
                        imports=unit.imports,
                        start_line=unit.start_line,
                        end_line=unit.end_line,
                    ))
            else:
                result.append(unit)

        return result


# Backwards-compatible alias — existing code importing CodeChunker still works
CodeChunker = ASTCodeChunker
