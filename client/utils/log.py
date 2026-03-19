from rich.table import Table
import time
from rich.console import Console
from rich.live import Live
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from classes.common import TestType

console = Console()

def render_ann_status(status, progress, index_task, eval_task):
    dataset = status["dataset"]
    meta = dataset["dataset"]
    index = dataset["indexProgress"]
    evalp = dataset["evaluationProgress"]

    # Index progress (vectors)
    if index["trainTotal"]:
        progress.update(index_task, total=index["trainTotal"], completed=index["vectorsProcessed"])
    else:
        progress.update(index_task, total=1, completed=0)

    # Evaluation progress (tests)
    if evalp["testTotal"]:
        progress.update(eval_task, total=evalp["testTotal"], completed=evalp["vectorsProcessed"])
    else:
        progress.update(eval_task, total=1, completed=0)

    table = Table()
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    table.add_row("Dataset", meta["name"])
    table.add_row("Overall State", status["overallState"])
    table.add_row("Phase", status["currentPhase"])
    table.add_row("Index Phase", index["phase"])
    table.add_row("Index Source", index["source"])
    table.add_row("Index State", index["state"])
    table.add_row("Eval State", evalp["state"])

    return table


def render_task_status(status, progress, index_task, eval_task):
    task = status["task"]
    meta = task["task"]
    index = task["indexProgress"]
    evalp = task["evaluationProgress"]

    # Index progress (documents)
    if index["documentsTotal"] is not None:
        progress.update(index_task, total=index["documentsTotal"], completed=index["documentsProcessed"])
    else:
        progress.update(index_task, total=1, completed=0)

    # Evaluation progress (questions)
    if evalp["questionsTotal"] is not None:
        progress.update(eval_task, total=evalp["questionsTotal"], completed=evalp["questionsProcessed"])
    else:
        progress.update(eval_task, total=1, completed=0)

    table = Table()
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    table.add_row("Task", meta["name"])
    table.add_row("Overall State", status["overallState"])
    table.add_row("Phase", status["currentPhase"])
    table.add_row("Task State", task["state"])
    table.add_row("Index Phase", index["phase"])
    table.add_row("Index Source", index["source"])
    table.add_row("Index State", index["state"])
    table.add_row("Chunks", str(index["chunksProcessed"]))
    table.add_row("Eval State", evalp["state"])

    return table

def render_status(status, test_type, progress, index_task, eval_task):
    if test_type == TestType.ANN:
        return render_ann_status(status, progress, index_task, eval_task)
    elif test_type == TestType.TASK:
        return render_task_status(status, progress, index_task, eval_task)
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

def display_status(status_stream, test_type):
    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    )

    index_task = progress.add_task("Indexing", total=1)
    eval_task = progress.add_task("Evaluating", total=1)

    with Live(refresh_per_second=10) as live:
        for status in status_stream:
            if status is None:
                continue
            table = render_status(status, test_type, progress, index_task, eval_task)

            grid = Table.grid(expand=True)
            grid.add_row(table)
            grid.add_row(progress)

            live.update(grid)
            time.sleep(0.5)
