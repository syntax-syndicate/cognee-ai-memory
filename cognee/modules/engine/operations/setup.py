from cognee.infrastructure.databases.relational import (
    create_db_and_tables as create_relational_db_and_tables,
)
from cognee.infrastructure.databases.vector.pgvector import (
    create_db_and_tables as create_pgvector_db_and_tables,
)
from cognee.modules.instrumentation.operations import setup_instrumentation


async def setup():
    await create_relational_db_and_tables()
    await create_pgvector_db_and_tables()
    setup_instrumentation()
