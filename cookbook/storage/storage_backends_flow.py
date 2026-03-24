"""
Cookbook: Configuring flows with different storage backends
==========================================================

Water supports multiple storage backends out of the box.  This example
shows how to wire up each one.

Backends available:
  - InMemoryStorage   (built-in, no dependencies)
  - SQLiteStorage     (built-in, uses stdlib sqlite3)
  - RedisStorage      (requires ``pip install redis``)
  - PostgresStorage   (requires ``pip install asyncpg``)
"""

import asyncio
from pydantic import BaseModel
from water.core import Flow, create_task
from water.storage import InMemoryStorage, SQLiteStorage


# ---------------------------------------------------------------------------
# 1. Define some simple tasks
# ---------------------------------------------------------------------------

class GreetInput(BaseModel):
    name: str = "World"

class GreetOutput(BaseModel):
    message: str

class ShoutOutput(BaseModel):
    message: str

greet = create_task(
    id="greet",
    input_schema=GreetInput,
    output_schema=GreetOutput,
    execute=lambda data, ctx: {"message": f"Hello, {data['input_data']['name']}!"},
)

shout = create_task(
    id="shout",
    input_schema=GreetOutput,
    output_schema=ShoutOutput,
    execute=lambda data, ctx: {"message": data["input_data"]["message"].upper()},
)


# ---------------------------------------------------------------------------
# 2. In-memory storage (default, good for tests / ephemeral runs)
# ---------------------------------------------------------------------------

memory_flow = Flow(
    id="memory_flow",
    storage=InMemoryStorage(),
)
memory_flow.then(greet).then(shout).register()


# ---------------------------------------------------------------------------
# 3. SQLite storage (built-in, persists to a local file)
# ---------------------------------------------------------------------------

sqlite_flow = Flow(
    id="sqlite_flow",
    storage=SQLiteStorage(db_path="my_flows.db"),
)
sqlite_flow.then(greet).then(shout).register()


# ---------------------------------------------------------------------------
# 4. Redis storage (requires `pip install redis`)
# ---------------------------------------------------------------------------

def make_redis_flow() -> Flow:
    from water.storage import RedisStorage

    redis_flow = Flow(
        id="redis_flow",
        storage=RedisStorage(
            redis_url="redis://localhost:6379",
            prefix="myapp",
        ),
    )
    redis_flow.then(greet).then(shout).register()
    return redis_flow


# ---------------------------------------------------------------------------
# 5. PostgreSQL storage (requires `pip install asyncpg`)
# ---------------------------------------------------------------------------

async def make_postgres_flow() -> Flow:
    from water.storage import PostgresStorage

    storage = PostgresStorage(dsn="postgresql://user:password@localhost:5432/mydb")
    # IMPORTANT: call initialize() once to create the tables
    await storage.initialize()

    pg_flow = Flow(
        id="postgres_flow",
        storage=storage,
    )
    pg_flow.then(greet).then(shout).register()
    return pg_flow


# ---------------------------------------------------------------------------
# Run the in-memory example
# ---------------------------------------------------------------------------

async def main():
    # In-memory example (always works)
    result = await memory_flow.run({"name": "Water"})
    print("InMemory result:", result)

    # SQLite example (always works)
    result = await sqlite_flow.run({"name": "Water"})
    print("SQLite result:", result)

    # Uncomment the following to try Redis (needs a running Redis server):
    # redis_flow = make_redis_flow()
    # result = await redis_flow.run({"name": "Water"})
    # print("Redis result:", result)

    # Uncomment the following to try PostgreSQL (needs a running PG server):
    # pg_flow = await make_postgres_flow()
    # result = await pg_flow.run({"name": "Water"})
    # print("Postgres result:", result)


if __name__ == "__main__":
    asyncio.run(main())
