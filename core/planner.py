"""
Planner - TODO-based planning system.
Inspired by Claude Code's TodoWrite tool.
Creates structured task lists with IDs, content, status, and priority.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import json
import time


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskPriority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Task:
    """A single task in the plan."""
    id: str
    content: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    result: Optional[str] = None
    subtasks: List['Task'] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "status": self.status.value,
            "priority": self.priority.value,
            "result": self.result,
            "subtasks": [t.to_dict() for t in self.subtasks],
        }


class Planner:
    """
    TODO-based planning system.
    
    The agent creates a plan before complex tasks, then updates it as work progresses.
    Like Claude Code, the current TODO state is injected as a system message
    after tool uses to keep the model focused.
    """

    def __init__(self):
        self._tasks: List[Task] = []
        self._task_counter = 0

    def create_plan(self, tasks: List[Dict[str, Any]]) -> str:
        """
        Create a new plan from a list of task descriptions.
        
        Args:
            tasks: List of {"content": str, "priority": str} dicts
        """
        self._tasks = []
        for task_data in tasks:
            self._task_counter += 1
            task = Task(
                id=f"task_{self._task_counter}",
                content=task_data["content"],
                priority=TaskPriority(task_data.get("priority", "medium")),
            )
            self._tasks.append(task)

        return self.get_plan_display()

    def update_task(self, task_id: str, status: str, result: Optional[str] = None) -> str:
        """Update a task's status."""
        task = self._find_task(task_id)
        if task is None:
            return f"Task not found: {task_id}"

        task.status = TaskStatus(status)
        task.result = result
        if status == "completed":
            task.completed_at = time.time()

        return self.get_plan_display()

    def get_current_task(self) -> Optional[Task]:
        """Get the next pending or in-progress task."""
        for task in self._tasks:
            if task.status == TaskStatus.IN_PROGRESS:
                return task
        for task in self._tasks:
            if task.status == TaskStatus.PENDING:
                return task
        return None

    def get_plan_display(self) -> str:
        """Get a human-readable plan display for injection into context."""
        if not self._tasks:
            return "No active plan."

        lines = ["📋 Current Plan:"]
        for task in self._tasks:
            icon = {
                TaskStatus.PENDING: "⬜",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.SKIPPED: "⏭️",
            }[task.status]
            priority_marker = "!" if task.priority == TaskPriority.HIGH else ""
            lines.append(f"  {icon} [{task.id}] {priority_marker}{task.content}")
            if task.result:
                lines.append(f"       → {task.result[:100]}")

        completed = sum(1 for t in self._tasks if t.status == TaskStatus.COMPLETED)
        total = len(self._tasks)
        lines.append(f"\n  Progress: {completed}/{total} completed")

        return "\n".join(lines)

    def is_complete(self) -> bool:
        """Check if all tasks are done (completed, failed, or skipped)."""
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED)
            for t in self._tasks
        )

    def _find_task(self, task_id: str) -> Optional[Task]:
        """Find a task by ID."""
        for task in self._tasks:
            if task.id == task_id:
                return task
        return None

    def to_dict(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self._tasks]
