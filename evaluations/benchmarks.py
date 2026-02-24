"""
Benchmark definitions for Code-SSG evaluation.

Supports standard code generation benchmarks:
- MBPP / MBPP-ET
- HumanEval / HumanEval-ET
- DS-1000
- CodeContests

Each benchmark defines prompts, test cases, and evaluation criteria.
"""

import json
import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkTask:
    """A single benchmark task."""
    task_id: str
    prompt: str
    function_name: str
    test_cases: List[str]
    entry_point: Optional[str] = None
    canonical_solution: Optional[str] = None
    difficulty: str = "medium"


@dataclass
class BenchmarkResult:
    """Result for a single task."""
    task_id: str
    passed: bool
    code: str
    execution_time_s: float
    error: Optional[str] = None
    ssg_report: Optional[Dict] = None


@dataclass
class BenchmarkSuiteResult:
    """Aggregate results for a benchmark suite."""
    benchmark_name: str
    total_tasks: int
    passed_tasks: int
    failed_tasks: int
    accuracy: float
    avg_execution_time_s: float
    results: List[BenchmarkResult] = field(default_factory=list)

    # For multi-trial statistics
    trial_accuracies: List[float] = field(default_factory=list)
    accuracy_std: float = 0.0
    accuracy_mean: float = 0.0


class BenchmarkRegistry:
    """Registry of available benchmarks."""

    _benchmarks: Dict[str, "BaseBenchmark"] = {}

    @classmethod
    def register(cls, name: str, benchmark: "BaseBenchmark"):
        cls._benchmarks[name] = benchmark

    @classmethod
    def get(cls, name: str) -> "BaseBenchmark":
        if name not in cls._benchmarks:
            raise ValueError(
                f"Unknown benchmark: {name}. "
                f"Available: {list(cls._benchmarks.keys())}"
            )
        return cls._benchmarks[name]

    @classmethod
    def list_benchmarks(cls) -> List[str]:
        return list(cls._benchmarks.keys())


class BaseBenchmark:
    """Base class for benchmarks."""

    def __init__(self, name: str, data_dir: str = "data"):
        self.name = name
        self.data_dir = Path(data_dir)
        self.tasks: List[BenchmarkTask] = []

    def load_tasks(self):
        """Load benchmark tasks from data directory."""
        raise NotImplementedError

    def evaluate_solution(
        self, task: BenchmarkTask, code: str, timeout: int = 10
    ) -> BenchmarkResult:
        """Evaluate a generated solution against test cases."""
        import subprocess
        import time

        start = time.time()
        # Build full test script
        test_script = code + "\n\n" + "\n".join(task.test_cases)

        try:
            result = subprocess.run(
                ["python3", "-c", test_script],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            passed = result.returncode == 0
            error = result.stderr if not passed else None
        except subprocess.TimeoutExpired:
            passed = False
            error = "timeout"
        except Exception as e:
            passed = False
            error = str(e)

        return BenchmarkResult(
            task_id=task.task_id,
            passed=passed,
            code=code,
            execution_time_s=time.time() - start,
            error=error,
        )

    def run_evaluation(
        self,
        generator_fn: Callable,
        ssg_validator=None,
        n_trials: int = 1,
    ) -> BenchmarkSuiteResult:
        """Run full benchmark evaluation with optional multi-trial."""
        if not self.tasks:
            self.load_tasks()

        all_trial_results = []

        for trial in range(n_trials):
            results = []
            for task in self.tasks:
                # Generate solution
                code = generator_fn(task.prompt)

                # SSG validation if available
                ssg_report = None
                if ssg_validator:
                    ssg_report = ssg_validator.validate_code_block(code)
                    ssg_report = {
                        "pass_rate": ssg_report.pass_rate,
                        "avg_confidence": ssg_report.avg_confidence,
                    }

                # Evaluate against test cases
                result = self.evaluate_solution(task, code)
                result.ssg_report = ssg_report
                results.append(result)

            all_trial_results.append(results)

        # Compute aggregate statistics
        trial_accuracies = []
        for trial_results in all_trial_results:
            passed = sum(1 for r in trial_results if r.passed)
            trial_accuracies.append(passed / len(trial_results))

        # Use last trial's detailed results
        final_results = all_trial_results[-1]
        passed_count = sum(1 for r in final_results if r.passed)
        avg_time = sum(r.execution_time_s for r in final_results) / len(final_results)

        import numpy as np
        return BenchmarkSuiteResult(
            benchmark_name=self.name,
            total_tasks=len(self.tasks),
            passed_tasks=passed_count,
            failed_tasks=len(self.tasks) - passed_count,
            accuracy=passed_count / len(self.tasks),
            avg_execution_time_s=avg_time,
            results=final_results,
            trial_accuracies=trial_accuracies,
            accuracy_mean=float(np.mean(trial_accuracies)),
            accuracy_std=float(np.std(trial_accuracies)),
        )


class MBPPBenchmark(BaseBenchmark):
    """MBPP (Mostly Basic Python Problems) benchmark."""

    def __init__(self, variant: str = "standard", data_dir: str = "data"):
        name = f"MBPP{'_ET' if variant == 'et' else ''}"
        super().__init__(name, data_dir)
        self.variant = variant

    def load_tasks(self):
        """Load MBPP tasks."""
        data_file = self.data_dir / f"mbpp{'_et' if self.variant == 'et' else ''}.jsonl"

        if not data_file.exists():
            logger.warning(f"Data file not found: {data_file}. Using sample tasks.")
            self._load_sample_tasks()
            return

        with open(data_file) as f:
            for line in f:
                data = json.loads(line)
                self.tasks.append(BenchmarkTask(
                    task_id=str(data["task_id"]),
                    prompt=data["text"],
                    function_name=data.get("function_name", ""),
                    test_cases=data.get("test_list", []),
                    canonical_solution=data.get("code", ""),
                ))

        logger.info(f"Loaded {len(self.tasks)} MBPP tasks")

    def _load_sample_tasks(self):
        """Load sample tasks for testing."""
        self.tasks = [
            BenchmarkTask(
                task_id="mbpp_1",
                prompt="Write a function to find the minimum cost path of a triangle.",
                function_name="min_cost",
                test_cases=[
                    "assert min_cost([[2], [3, 4], [6, 5, 7], [4, 1, 8, 3]]) == 11",
                ],
            ),
            BenchmarkTask(
                task_id="mbpp_2",
                prompt="Write a python function to check whether every even index contains even numbers.",
                function_name="even_position",
                test_cases=[
                    "assert even_position([3,2,1]) == False",
                    "assert even_position([2,1,4]) == True",
                ],
            ),
            BenchmarkTask(
                task_id="mbpp_3",
                prompt="Write a function to find the maximum sum of subsequence with no adjacent elements.",
                function_name="max_sum_subseq",
                test_cases=[
                    "assert max_sum_subseq([1, 2, 9, 4, 5, 0, 4, 11, 6]) == 26",
                ],
            ),
        ]


class HumanEvalBenchmark(BaseBenchmark):
    """HumanEval benchmark."""

    def __init__(self, variant: str = "standard", data_dir: str = "data"):
        name = f"HumanEval{'_ET' if variant == 'et' else ''}"
        super().__init__(name, data_dir)
        self.variant = variant

    def load_tasks(self):
        data_file = self.data_dir / f"humaneval{'_et' if self.variant == 'et' else ''}.jsonl"

        if not data_file.exists():
            logger.warning(f"Data file not found: {data_file}. Using sample tasks.")
            self._load_sample_tasks()
            return

        with open(data_file) as f:
            for line in f:
                data = json.loads(line)
                self.tasks.append(BenchmarkTask(
                    task_id=data["task_id"],
                    prompt=data["prompt"],
                    function_name=data.get("entry_point", ""),
                    test_cases=data.get("test", "").split("\n"),
                    entry_point=data.get("entry_point"),
                    canonical_solution=data.get("canonical_solution"),
                ))

    def _load_sample_tasks(self):
        self.tasks = [
            BenchmarkTask(
                task_id="HumanEval/0",
                prompt='from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
                function_name="has_close_elements",
                test_cases=[
                    "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True",
                    "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False",
                ],
            ),
        ]


class DS1000Benchmark(BaseBenchmark):
    """DS-1000 data science benchmark."""

    def __init__(self, data_dir: str = "data"):
        super().__init__("DS-1000", data_dir)

    def load_tasks(self):
        data_file = self.data_dir / "ds1000.jsonl"
        if not data_file.exists():
            logger.warning("DS-1000 data not found. Using placeholder.")
            return
        # Load logic similar to MBPP...


class CodeContestsBenchmark(BaseBenchmark):
    """CodeContests competitive programming benchmark."""

    def __init__(self, data_dir: str = "data"):
        super().__init__("CodeContests", data_dir)

    def load_tasks(self):
        data_file = self.data_dir / "codecontests.jsonl"
        if not data_file.exists():
            logger.warning("CodeContests data not found. Using placeholder.")
            return


# Register all benchmarks
BenchmarkRegistry.register("mbpp", MBPPBenchmark("standard"))
BenchmarkRegistry.register("mbpp_et", MBPPBenchmark("et"))
BenchmarkRegistry.register("humaneval", HumanEvalBenchmark("standard"))
BenchmarkRegistry.register("humaneval_et", HumanEvalBenchmark("et"))
BenchmarkRegistry.register("ds1000", DS1000Benchmark())
BenchmarkRegistry.register("codecontests", CodeContestsBenchmark())
