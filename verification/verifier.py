"""
Scientific Verification Engine.
Inspired by EG-CFG's line-by-line execution-guided code generation.

Core idea: Transform "execute each line of code" into "scientifically verify each statement"
- Code execution verifier: Run code, trace execution, check results
- LLM judgment verifier: Ask LLM to verify scientific correctness
- Hybrid: Combine both for maximum accuracy
"""

import sys
import os
import ast
import traceback
import subprocess
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class VerificationMode(Enum):
    EXECUTION = "execution"      # Run code, check output
    LLM_JUDGE = "llm_judge"      # LLM evaluates correctness
    HYBRID = "hybrid"            # Both execution + LLM
    TRACE = "trace"              # Line-by-line execution trace


class VerificationResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"
    ERROR = "error"


@dataclass
class StatementVerification:
    """Verification result for a single statement/line."""
    line_number: int
    statement: str
    result: VerificationResult = VerificationResult.SKIP
    details: str = ""
    execution_trace: Optional[str] = None
    llm_judgment: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class VerificationReport:
    """Complete verification report for a code block."""
    file_path: str
    total_statements: int
    verified: int = 0
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    errors: int = 0
    statements: List[StatementVerification] = field(default_factory=list)
    execution_time_ms: float = 0.0
    mode: VerificationMode = VerificationMode.EXECUTION

    def is_success(self) -> bool:
        return self.failed == 0 and self.errors == 0

    def to_display(self) -> str:
        """Format for display."""
        status = "✅ PASSED" if self.is_success() else "❌ FAILED"
        lines = [
            f"\n{'='*60}",
            f"Verification Report: {self.file_path}",
            f"{'='*60}",
            f"Status: {status}",
            f"Mode: {self.mode.value}",
            f"Statements: {self.total_statements} total, {self.passed} passed, "
            f"{self.failed} failed, {self.warnings} warnings, {self.errors} errors",
            f"Time: {self.execution_time_ms:.0f}ms",
            f"{'='*60}",
        ]

        for sv in self.statements:
            icon = {
                VerificationResult.PASS: "✅",
                VerificationResult.FAIL: "❌",
                VerificationResult.WARNING: "⚠️",
                VerificationResult.SKIP: "⏭️",
                VerificationResult.ERROR: "💥",
            }[sv.result]
            lines.append(f"\n{icon} Line {sv.line_number}: {sv.statement[:80]}")
            if sv.details:
                lines.append(f"   {sv.details}")
            if sv.execution_trace:
                lines.append(f"   Trace: {sv.execution_trace[:200]}")
            if sv.variables:
                vars_str = ", ".join(f"{k}={v}" for k, v in list(sv.variables.items())[:5])
                lines.append(f"   Vars: {vars_str}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "success": self.is_success(),
            "total_statements": self.total_statements,
            "passed": self.passed,
            "failed": self.failed,
            "mode": self.mode.value,
            "statements": [
                {
                    "line": s.line_number,
                    "statement": s.statement,
                    "result": s.result.value,
                    "details": s.details,
                }
                for s in self.statements
            ],
        }


class ExecutionVerifier:
    """
    Verify code by executing it and tracing each statement.
    Inspired by EG-CFG's trepan-xpy debugger for execution traces.
    Uses Python's sys.settrace for line-by-line tracing.
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def verify(self, code: str, file_path: str = "<code>",
               test_cases: Optional[List[str]] = None) -> VerificationReport:
        """Verify code by execution."""
        start = time.time()
        report = VerificationReport(
            file_path=file_path,
            total_statements=0,
            mode=VerificationMode.EXECUTION,
        )

        # Parse to get statement count
        try:
            tree = ast.parse(code)
            statements = self._extract_statements(tree, code)
            report.total_statements = len(statements)
        except SyntaxError as e:
            report.errors = 1
            report.statements.append(StatementVerification(
                line_number=e.lineno or 0,
                statement=str(e),
                result=VerificationResult.ERROR,
                details=f"Syntax error: {e.msg}",
            ))
            report.execution_time_ms = (time.time() - start) * 1000
            return report

        # Execute with tracing
        trace_data = self._execute_with_trace(code, file_path)

        # Build verification results
        for stmt_line, stmt_text in statements:
            sv = StatementVerification(
                line_number=stmt_line,
                statement=stmt_text,
            )

            if stmt_line in trace_data.get("executed_lines", set()):
                if stmt_line in trace_data.get("error_lines", {}):
                    sv.result = VerificationResult.FAIL
                    sv.details = trace_data["error_lines"][stmt_line]
                    report.failed += 1
                else:
                    sv.result = VerificationResult.PASS
                    report.passed += 1
                    # Capture variables at this line
                    if stmt_line in trace_data.get("variables", {}):
                        sv.variables = trace_data["variables"][stmt_line]
            else:
                sv.result = VerificationResult.SKIP
                sv.details = "Line not reached during execution"

            sv.execution_trace = trace_data.get("traces", {}).get(stmt_line, "")
            report.statements.append(sv)
            report.verified += 1

        # Run test cases if provided
        if test_cases:
            self._run_test_cases(code, test_cases, report)

        # Handle global execution error
        if trace_data.get("global_error"):
            report.errors += 1
            report.statements.append(StatementVerification(
                line_number=0,
                statement="[Global Error]",
                result=VerificationResult.ERROR,
                details=trace_data["global_error"],
            ))

        report.execution_time_ms = (time.time() - start) * 1000
        return report

    def _execute_with_trace(self, code: str, file_path: str) -> Dict[str, Any]:
        """Execute code with sys.settrace to capture line-by-line execution."""
        trace_data = {
            "executed_lines": set(),
            "error_lines": {},
            "variables": {},
            "traces": {},
            "global_error": None,
        }

        # Write code to temp file and execute in subprocess for safety
        import tempfile

        indented_code = "\n".join("    " + line for line in code.split("\n"))

        # Build the tracer script as a list of lines so we can count precisely
        preamble_lines = [
            "import sys",
            "import json",
            "import traceback",
            "",
            "_trace_data = {",
            '    "executed_lines": [],',
            '    "variables": {},',
            '    "error_lines": {},',
            "}",
            "",
            "def _tracer(frame, event, arg):",
            "    if frame.f_code.co_filename == __file__:",
            "        if event == 'line':",
            "            line_no = frame.f_lineno",
            '            _trace_data["executed_lines"].append(line_no)',
            "            local_vars = {}",
            "            for k, v in frame.f_locals.items():",
            "                if not k.startswith('_'):",
            "                    try:",
            "                        local_vars[k] = repr(v)[:100]",
            "                    except:",
            '                        local_vars[k] = "<unprintable>"',
            '            _trace_data["variables"][str(line_no)] = local_vars',
            "        elif event == 'exception':",
            "            line_no = frame.f_lineno",
            '            _trace_data["error_lines"][str(line_no)] = str(arg[1])',
            "    return _tracer",
            "",
        ]

        # CODE_START_LINE = number of preamble lines + 3 more lines (the _CODE_START_LINE assignment, blank, sys.settrace, try:)
        # The user code indented lines start right after "try:"
        code_start = len(preamble_lines) + 4  # +4 for: _CODE_START_LINE=, blank, sys.settrace, try:

        script_lines = preamble_lines + [
            f"_CODE_START_LINE = {code_start}",
            "",
            "sys.settrace(_tracer)",
            "try:",
        ]

        # Add indented user code
        for line in code.split("\n"):
            script_lines.append("    " + line)

        script_lines += [
            "except Exception as e:",
            '    _trace_data["global_error"] = traceback.format_exc()',
            "finally:",
            "    sys.settrace(None)",
            "",
            "adjusted = {}",
            'for k, v in _trace_data["variables"].items():',
            "    adjusted[str(int(k) - _CODE_START_LINE + 1)] = v",
            '_trace_data["variables"] = adjusted',
            '_trace_data["executed_lines"] = [l - _CODE_START_LINE + 1 for l in _trace_data["executed_lines"]]',
            "adj_errors = {}",
            'for k, v in _trace_data["error_lines"].items():',
            "    adj_errors[str(int(k) - _CODE_START_LINE + 1)] = v",
            '_trace_data["error_lines"] = adj_errors',
            "",
            'print("__TRACE_START__")',
            "print(json.dumps(_trace_data, default=str))",
            'print("__TRACE_END__")',
        ]

        tracer_script = "\n".join(script_lines)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(tracer_script)
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True, text=True, timeout=self.timeout,
            )

            # Parse trace data from output
            output = result.stdout
            if "__TRACE_START__" in output and "__TRACE_END__" in output:
                json_str = output.split("__TRACE_START__")[1].split("__TRACE_END__")[0].strip()
                parsed = json.loads(json_str)
                trace_data["executed_lines"] = set(parsed.get("executed_lines", []))
                trace_data["variables"] = {
                    int(k): v for k, v in parsed.get("variables", {}).items()
                }
                trace_data["error_lines"] = {
                    int(k): v for k, v in parsed.get("error_lines", {}).items()
                }
                if parsed.get("global_error"):
                    trace_data["global_error"] = parsed["global_error"]

            if result.returncode != 0 and not trace_data["global_error"]:
                trace_data["global_error"] = result.stderr[:500]

        except subprocess.TimeoutExpired:
            trace_data["global_error"] = f"Execution timed out after {self.timeout}s"
        except Exception as e:
            trace_data["global_error"] = str(e)
        finally:
            os.unlink(temp_path)

        return trace_data

    def _run_test_cases(self, code: str, test_cases: List[str], report: VerificationReport):
        """Run test cases against the code (like EG-CFG's test evaluation)."""
        for i, test in enumerate(test_cases):
            full_code = code + "\n\n" + test
            try:
                result = subprocess.run(
                    [sys.executable, "-c", full_code],
                    capture_output=True, text=True, timeout=self.timeout,
                )
                if result.returncode == 0:
                    report.statements.append(StatementVerification(
                        line_number=0,
                        statement=f"Test {i+1}: {test[:60]}",
                        result=VerificationResult.PASS,
                        details="Test passed",
                    ))
                    report.passed += 1
                else:
                    report.statements.append(StatementVerification(
                        line_number=0,
                        statement=f"Test {i+1}: {test[:60]}",
                        result=VerificationResult.FAIL,
                        details=result.stderr[:200],
                    ))
                    report.failed += 1
            except Exception as e:
                report.statements.append(StatementVerification(
                    line_number=0,
                    statement=f"Test {i+1}: {test[:60]}",
                    result=VerificationResult.ERROR,
                    details=str(e),
                ))
                report.errors += 1

    @staticmethod
    def _extract_statements(tree: ast.AST, code: str) -> List[Tuple[int, str]]:
        """Extract (line_number, statement_text) pairs from AST."""
        lines = code.split('\n')
        statements = []
        for node in ast.walk(tree):
            if isinstance(node, ast.stmt) and hasattr(node, 'lineno'):
                line_text = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                if line_text and not line_text.startswith('#'):
                    statements.append((node.lineno, line_text))
        statements.sort(key=lambda x: x[0])
        return statements


class LLMVerifier:
    """
    Verify statements using LLM judgment.
    
    Key insight from EG-CFG: "LLM在判断一句话是否符合科学性的时候正确率奇高"
    (LLMs have extremely high accuracy when judging if a statement is scientifically sound)
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model

    def verify_statement(self, statement: str, context: str = "") -> StatementVerification:
        """Verify a single statement using LLM judgment."""
        prompt = self._build_prompt(statement, context)

        if not self.api_key:
            return StatementVerification(
                line_number=0,
                statement=statement,
                result=VerificationResult.SKIP,
                details="No API key configured for LLM verification",
                llm_judgment="skipped",
            )

        try:
            result = self._call_llm(prompt)
            judgment = self._parse_judgment(result)
            return StatementVerification(
                line_number=0,
                statement=statement,
                result=judgment["result"],
                details=judgment["explanation"],
                llm_judgment=result,
            )
        except Exception as e:
            return StatementVerification(
                line_number=0,
                statement=statement,
                result=VerificationResult.ERROR,
                details=f"LLM verification error: {e}",
            )

    def verify_code_block(self, code: str, file_path: str = "<code>") -> VerificationReport:
        """Verify all statements in a code block."""
        start = time.time()
        report = VerificationReport(
            file_path=file_path,
            total_statements=0,
            mode=VerificationMode.LLM_JUDGE,
        )

        lines = code.split('\n')
        statements = [(i+1, line.strip()) for i, line in enumerate(lines)
                       if line.strip() and not line.strip().startswith('#')]
        report.total_statements = len(statements)

        for line_num, stmt in statements:
            context = "\n".join(lines[max(0, line_num-5):line_num])
            sv = self.verify_statement(stmt, context)
            sv.line_number = line_num
            report.statements.append(sv)
            report.verified += 1

            if sv.result == VerificationResult.PASS:
                report.passed += 1
            elif sv.result == VerificationResult.FAIL:
                report.failed += 1
            elif sv.result == VerificationResult.WARNING:
                report.warnings += 1
            elif sv.result == VerificationResult.ERROR:
                report.errors += 1

        report.execution_time_ms = (time.time() - start) * 1000
        return report

    def _build_prompt(self, statement: str, context: str = "") -> str:
        return f"""Analyze this code statement for scientific/logical correctness.

Context (preceding code):
```
{context}
```

Statement to verify:
```
{statement}
```

Evaluate:
1. Is the statement syntactically correct?
2. Is the logic sound?
3. Are there potential bugs or edge cases?
4. Is it consistent with the context?

Respond with JSON:
{{"result": "pass"|"fail"|"warning", "explanation": "brief explanation"}}"""

    def _call_llm(self, prompt: str) -> str:
        """Call LLM API. Uses subprocess to call Anthropic API."""
        result = subprocess.run(
            [sys.executable, "-c", f"""
import json
try:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="{self.model}",
        max_tokens=500,
        messages=[{{"role": "user", "content": {json.dumps(prompt)}}}]
    )
    print(response.content[0].text)
except Exception as e:
    print(json.dumps({{"result": "error", "explanation": str(e)}}))
"""],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip()

    def _parse_judgment(self, result: str) -> Dict[str, Any]:
        """Parse LLM judgment response."""
        try:
            # Try to extract JSON from response
            if '{' in result:
                json_str = result[result.index('{'):result.rindex('}')+1]
                data = json.loads(json_str)
                result_map = {
                    "pass": VerificationResult.PASS,
                    "fail": VerificationResult.FAIL,
                    "warning": VerificationResult.WARNING,
                }
                return {
                    "result": result_map.get(data.get("result", ""), VerificationResult.WARNING),
                    "explanation": data.get("explanation", "No explanation"),
                }
        except (json.JSONDecodeError, ValueError):
            pass
        return {
            "result": VerificationResult.WARNING,
            "explanation": f"Could not parse LLM response: {result[:200]}",
        }


class HybridVerifier:
    """
    Combine execution and LLM verification for maximum accuracy.
    First runs code, then asks LLM to evaluate any ambiguous results.
    """

    def __init__(self, **kwargs):
        self.exec_verifier = ExecutionVerifier(**{k: v for k, v in kwargs.items()
                                                   if k in ('timeout',)})
        self.llm_verifier = LLMVerifier(**{k: v for k, v in kwargs.items()
                                            if k in ('api_key', 'model')})

    def verify(self, code: str, file_path: str = "<code>",
               test_cases: Optional[List[str]] = None) -> VerificationReport:
        """Run hybrid verification."""
        # Phase 1: Execution
        exec_report = self.exec_verifier.verify(code, file_path, test_cases)

        # Phase 2: LLM verification for failed/warning statements
        for sv in exec_report.statements:
            if sv.result in (VerificationResult.FAIL, VerificationResult.WARNING):
                llm_sv = self.llm_verifier.verify_statement(
                    sv.statement,
                    context=f"Execution result: {sv.details}"
                )
                sv.llm_judgment = llm_sv.llm_judgment
                # If LLM says it's actually OK, downgrade to warning
                if sv.result == VerificationResult.FAIL and llm_sv.result == VerificationResult.PASS:
                    sv.result = VerificationResult.WARNING
                    sv.details += " [LLM: likely correct despite execution issue]"

        # Recalculate counts
        exec_report.passed = sum(1 for s in exec_report.statements if s.result == VerificationResult.PASS)
        exec_report.failed = sum(1 for s in exec_report.statements if s.result == VerificationResult.FAIL)
        exec_report.warnings = sum(1 for s in exec_report.statements if s.result == VerificationResult.WARNING)
        exec_report.errors = sum(1 for s in exec_report.statements if s.result == VerificationResult.ERROR)
        exec_report.mode = VerificationMode.HYBRID

        return exec_report


def create_verifier(mode: VerificationMode = VerificationMode.EXECUTION,
                    **kwargs):
    """Factory function to create the appropriate verifier."""
    if mode == VerificationMode.EXECUTION or mode == VerificationMode.TRACE:
        return ExecutionVerifier(**{k: v for k, v in kwargs.items() if k in ('timeout',)})
    elif mode == VerificationMode.LLM_JUDGE:
        return LLMVerifier(**{k: v for k, v in kwargs.items() if k in ('api_key', 'model')})
    elif mode == VerificationMode.HYBRID:
        return HybridVerifier(**kwargs)
    else:
        raise ValueError(f"Unknown verification mode: {mode}")
