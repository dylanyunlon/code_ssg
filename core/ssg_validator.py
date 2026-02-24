"""
Scientific Statement Grounding (SSG) Validator.

Converts each line of code into a scientific statement and validates it
through either code execution, LLM-as-judge, or both (hybrid).

Inspired by EG-CFG's line-by-line execution-guided approach, but instead
of using execution traces for CFG guidance, we use scientific verification
to ground each code statement in verifiable truth.
"""

import ast
import json
import logging
import subprocess
import traceback
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a single scientific statement."""
    code_line: str
    statement: str
    valid: bool
    method: str  # "code_exec", "llm_judge", "hybrid"
    confidence: float  # 0.0 to 1.0
    execution_result: Optional[Dict] = None
    llm_judgment: Optional[Dict] = None
    error: Optional[str] = None


@dataclass
class SSGReport:
    """Aggregate report for SSG validation of a code block."""
    total_lines: int
    validated_lines: int
    passed_lines: int
    failed_lines: int
    skipped_lines: int  # Comments, blank lines, etc.
    avg_confidence: float
    results: List[ValidationResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if self.validated_lines == 0:
            return 0.0
        return self.passed_lines / self.validated_lines


class CodeToStatementConverter:
    """
    Converts Python code lines into verifiable scientific/logical statements.

    Each type of code construct maps to a specific kind of statement:
    - Assignment → "Variable X is assigned value Y of type T"
    - Function call → "Function F is called with arguments A, expected to return R"
    - Conditional → "Condition C evaluates to True/False given current state"
    - Loop → "Loop iterates over sequence S of length N"
    - Import → "Module M exists and provides functions [F1, F2, ...]"
    - Return → "Function returns value V of type T"
    """

    def convert(self, code_line: str, context: Optional[Dict] = None) -> Optional[str]:
        """Convert a single code line to a verifiable statement."""
        stripped = code_line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith("#"):
            return None

        try:
            tree = ast.parse(stripped)
            if not tree.body:
                return None
            node = tree.body[0]
            return self._node_to_statement(node, stripped, context)
        except SyntaxError:
            # Partial code (e.g., continuation) - use heuristic
            return self._heuristic_statement(stripped, context)

    def _node_to_statement(
        self, node: ast.AST, raw: str, context: Optional[Dict]
    ) -> str:
        """Convert an AST node to a statement."""

        if isinstance(node, ast.Assign):
            targets = ", ".join(ast.dump(t) for t in node.targets)
            return (
                f"Assignment: The expression '{raw}' assigns a value to "
                f"variable(s). The right-hand side expression should produce "
                f"a valid value without errors."
            )

        elif isinstance(node, ast.AugAssign):
            return (
                f"Augmented assignment: '{raw}' modifies an existing variable "
                f"using an arithmetic/logical operator."
            )

        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func_name = ast.dump(node.value.func)
            n_args = len(node.value.args) + len(node.value.keywords)
            return (
                f"Function call: '{raw}' calls a function with {n_args} "
                f"argument(s). The function should be defined and callable."
            )

        elif isinstance(node, ast.If):
            return (
                f"Conditional: '{raw}' evaluates a boolean condition. "
                f"The condition expression should be valid and evaluate to a boolean."
            )

        elif isinstance(node, ast.For):
            return (
                f"For loop: '{raw}' iterates over an iterable. "
                f"The iterable should be a valid sequence/iterator."
            )

        elif isinstance(node, ast.While):
            return (
                f"While loop: '{raw}' continues while a condition is True. "
                f"The condition should eventually become False to avoid infinite loop."
            )

        elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            return (
                f"Import: '{raw}' imports a module or names from a module. "
                f"The module should exist and be importable."
            )

        elif isinstance(node, ast.Return):
            return (
                f"Return: '{raw}' returns a value from a function. "
                f"The return value expression should be valid."
            )

        elif isinstance(node, ast.FunctionDef):
            n_params = len(node.args.args)
            return (
                f"Function definition: '{node.name}' is defined with "
                f"{n_params} parameter(s). The function body should be valid Python."
            )

        elif isinstance(node, ast.ClassDef):
            bases = [ast.dump(b) for b in node.bases]
            return (
                f"Class definition: '{node.name}' is defined"
                + (f" inheriting from {', '.join(bases)}" if bases else "")
                + ". The class body should contain valid methods and attributes."
            )

        else:
            return f"Statement: '{raw}' is a valid Python statement that executes without errors."

    def _heuristic_statement(self, raw: str, context: Optional[Dict]) -> str:
        """Heuristic fallback for unparseable lines."""
        if "=" in raw and not raw.startswith("=="):
            return f"Assignment or comparison: '{raw}' performs an operation."
        elif raw.endswith(":"):
            return f"Block header: '{raw}' starts a new code block."
        else:
            return f"Expression: '{raw}' evaluates to a value or performs a side effect."


class CodeExecutionValidator:
    """Validates statements by actually executing code."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def validate(
        self, code_line: str, full_context: str = "", test_code: str = ""
    ) -> Dict[str, Any]:
        """
        Validate a code line by executing it in context.

        Args:
            code_line: The specific line to validate
            full_context: Full code context up to this line
            test_code: Optional test code to run after

        Returns:
            Dict with 'valid', 'output', 'error' keys
        """
        # Build the execution script
        script = full_context
        if test_code:
            script += "\n" + test_code

        try:
            result = subprocess.run(
                ["python3", "-c", script],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            return {
                "valid": result.returncode == 0,
                "stdout": result.stdout[-2000:],
                "stderr": result.stderr[-2000:],
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"valid": False, "error": "execution timeout"}
        except Exception as e:
            return {"valid": False, "error": str(e)}


class LLMJudgeValidator:
    """
    Validates statements using an LLM as judge.

    Research shows that LLMs have extremely high accuracy when judging
    whether a single statement is scientifically/logically correct.
    This exploits that strength for code validation.
    """

    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
        self.model = model

    def validate(
        self, code_line: str, statement: str, context: str = ""
    ) -> Dict[str, Any]:
        """
        Ask the LLM to judge if a statement is valid.

        The prompt is designed to get a binary yes/no with confidence.
        """
        prompt = f"""You are a code validation judge. Given a Python code line and its
derived scientific statement, determine if the statement is accurate.

Code line: {code_line}
Statement: {statement}
Context (preceding code):
{context[-2000:] if context else 'None'}

Respond in JSON format:
{{
    "valid": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Focus on:
1. Is the statement logically correct about what the code does?
2. Would the code execute without errors in the given context?
3. Are there any type errors, undefined variables, or logic bugs?"""

        # In production, call the LLM API
        # response = anthropic_client.messages.create(...)
        logger.debug(f"LLM judge query for: {code_line[:80]}...")

        # Placeholder
        return {
            "valid": True,
            "confidence": 0.85,
            "reasoning": "placeholder - connect LLM API",
        }


class SSGValidator:
    """
    Main SSG Validator that orchestrates code-to-statement conversion
    and validation through multiple methods.

    Modes:
    - "code_exec": Validate by executing code
    - "llm_judge": Validate by asking an LLM
    - "hybrid": Use both methods (AND logic for strictness)
    """

    def __init__(
        self,
        mode: str = "hybrid",
        model: str = "claude-sonnet-4-5-20250929",
        exec_timeout: int = 10,
    ):
        self.mode = mode
        self.converter = CodeToStatementConverter()
        self.exec_validator = CodeExecutionValidator(timeout=exec_timeout)
        self.llm_validator = LLMJudgeValidator(model=model)

    def validate(
        self,
        code_line: str,
        statement: Optional[str] = None,
        method: Optional[str] = None,
        context: str = "",
    ) -> Dict[str, Any]:
        """
        Validate a single code line's scientific statement.

        If no statement is provided, auto-generates one from the code.
        """
        method = method or self.mode

        # Auto-generate statement if not provided
        if statement is None:
            statement = self.converter.convert(code_line)
            if statement is None:
                return {
                    "valid": True,
                    "skipped": True,
                    "reason": "non-code line (comment/blank)",
                }

        result = ValidationResult(
            code_line=code_line,
            statement=statement,
            valid=True,
            method=method,
            confidence=0.0,
        )

        # Code execution validation
        if method in ("code_exec", "hybrid"):
            exec_result = self.exec_validator.validate(code_line, context)
            result.execution_result = exec_result
            if not exec_result.get("valid", False):
                result.valid = False

        # LLM judge validation
        if method in ("llm_judge", "hybrid"):
            llm_result = self.llm_validator.validate(code_line, statement, context)
            result.llm_judgment = llm_result
            if not llm_result.get("valid", False):
                result.valid = False
            result.confidence = llm_result.get("confidence", 0.0)

        return {
            "valid": result.valid,
            "statement": result.statement,
            "confidence": result.confidence,
            "method": result.method,
            "execution_result": result.execution_result,
            "llm_judgment": result.llm_judgment,
        }

    def validate_code_block(
        self, code: str, method: Optional[str] = None
    ) -> SSGReport:
        """
        Validate an entire code block line by line.

        This is the EG-CFG-inspired pipeline: process each line,
        convert to statement, validate, and accumulate results.
        """
        lines = code.split("\n")
        results = []
        context_so_far = ""
        validated = 0
        passed = 0
        failed = 0
        skipped = 0

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                skipped += 1
                continue

            statement = self.converter.convert(line)
            if statement is None:
                skipped += 1
                continue

            validated += 1
            result = self.validate(
                code_line=line,
                statement=statement,
                method=method,
                context=context_so_far,
            )

            if result.get("valid"):
                passed += 1
            else:
                failed += 1

            results.append(ValidationResult(
                code_line=line,
                statement=statement,
                valid=result.get("valid", False),
                method=result.get("method", self.mode),
                confidence=result.get("confidence", 0.0),
                execution_result=result.get("execution_result"),
                llm_judgment=result.get("llm_judgment"),
            ))

            context_so_far += line + "\n"

        avg_conf = (
            sum(r.confidence for r in results) / len(results)
            if results else 0.0
        )

        return SSGReport(
            total_lines=len(lines),
            validated_lines=validated,
            passed_lines=passed,
            failed_lines=failed,
            skipped_lines=skipped,
            avg_confidence=avg_conf,
            results=results,
        )
