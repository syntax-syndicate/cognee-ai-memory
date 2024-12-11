import inspect
import json
import logging

from cognee.modules.settings import get_current_settings
from cognee.modules.users.methods import get_default_user
from cognee.modules.pipelines.operations.run_tasks_base import run_tasks_base
from cognee.shared.utils import send_telemetry

from ..tasks.Task import Task

logger = logging.getLogger("run_tasks(tasks: [Task], data)")


async def run_tasks_with_telemetry(tasks: list[Task], data, pipeline_name: str):

    config = get_current_settings()
    
    logger.debug("\nRunning pipeline with configuration:\n%s\n", json.dumps(config, indent = 1))
    
    user = await get_default_user()
    
    try:
        logger.info("Pipeline run started: `%s`", pipeline_name)
        send_telemetry("Pipeline Run Started", 
                       user.id, 
                       additional_properties = {"pipeline_name": pipeline_name, } | config
                       )
        
        async for result in run_tasks_base(tasks, data, user):
            yield result

        logger.info("Pipeline run completed: `%s`", pipeline_name)
        send_telemetry("Pipeline Run Completed", 
                       user.id, 
                       additional_properties = {"pipeline_name": pipeline_name, }
                       )
    except Exception as error:
        logger.error(
            "Pipeline run errored: `%s`\n%s\n",
            pipeline_name,
            str(error),
            exc_info = True,
        )
        send_telemetry("Pipeline Run Errored", 
                       user.id, 
                       additional_properties = {"pipeline_name": pipeline_name, } | config
                       )

        raise error

async def run_tasks(tasks: list[Task], data = None, pipeline_name: str = "default_pipeline"):
    
    async for result in run_tasks_with_telemetry(tasks, data, pipeline_name):
        yield result
