import inspect
import logging
from typing import Any, AsyncGenerator, AsyncIterator, List

from cognee.modules.users.models import User
from cognee.shared.utils import send_telemetry

from ..tasks.Task import Task

logger = logging.getLogger(__name__)


async def run_tasks_base(
        tasks: List[Task],
        data: Any = None,
        user: User = None
) -> AsyncGenerator[Any, None]:
    """Recursively chains tasks, passing and accumulating data sequentially to produce the output of the final task."""
    if not tasks:
        yield data
        return

    args = [data] if data is not None else []
    current_task, remaining_tasks = tasks[0], tasks[1:]
    batch_size = remaining_tasks[0].task_config["batch_size"] if remaining_tasks else 1

    current_task_type, current_results_iterator = await get_task_type_and_results_iterator(current_task, args)
    log_task_phase(current_task, current_task_type, user, phase="Started")
    try:
        async for batch in get_next_batch(current_results_iterator, batch_size):
            batch_data = batch[0] if len(batch) == 1 else batch
            async for remaining_result in run_tasks_base(remaining_tasks, batch_data, user):
                yield remaining_result

        log_task_phase(current_task, current_task_type, user, phase="Completed")
    except Exception as e:
        log_task_phase(current_task, current_task_type, user, phase="Errored", error=e)


async def get_task_type_and_results_iterator(
        task: Task,
        args: List[Any]
) -> (str, AsyncIterator[Any]):
    """Determines the task type and returns the task type and an async iterator for its results."""
    if inspect.isasyncgenfunction(task.executable):
        return "Async Generator", task.run(*args)

    if inspect.isgeneratorfunction(task.executable):
        return "Generator", wrap_sync_iter(task.run(*args))

    if inspect.iscoroutinefunction(task.executable):
        return "Coroutine", wrap_sync_iter([await task.run(*args)])

    return "Function", wrap_sync_iter([task.run(*args)])


async def get_next_batch(
        results: AsyncIterator[Any],
        batch_size: int
) -> AsyncIterator[List[Any]]:
    """Yields batches of results of a specified size."""
    batch = []
    async for result in results:
        batch.append(result)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # Yield the last batch if not empty
        yield batch


async def wrap_sync_iter(iterable: Any) -> AsyncGenerator[Any, None]:
    """Wraps a synchronous iterable into an async generator."""
    for item in iterable:
        yield item


def log_task_phase(task: Task, task_type: str, user: User, phase: str, error: Exception = None) -> None:
    """Logs and sends telemetry for a specific phase of a task's lifecycle (start, complete, error)."""
    event_name = f"{task_type} Task {phase}"

    if error:
        logger.error("%s: `%s`\n%s\n", event_name, task.executable.__name__, str(error), exc_info=True)
        send_telemetry(event_name, user.id, {"task_name": task.executable.__name__})
        raise error

    logger.info("%s: `%s`", event_name, task.executable.__name__)
    send_telemetry(event_name, user.id, {"task_name": task.executable.__name__})

