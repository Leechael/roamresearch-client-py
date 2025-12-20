from typing import List, cast
import pprint
from itertools import chain
import logging
import signal
import os
import asyncio
import uuid
from pathlib import Path
import traceback
import sqlite3
from datetime import datetime

from dotenv import load_dotenv
import mcp.types as types
from mcp.server.fastmcp import FastMCP
import httpx
import pendulum

from .client import RoamClient, create_page, create_block
from .formatter import format_block
from .gfm_to_roam import gfm_to_batch_actions


class CancelledErrorFilter(logging.Filter):
    def filter(self, record):
        return "asyncio.exceptions.CancelledError" not in record.getMessage()

for logger_name in ("uvicorn.error", "uvicorn.access", "uvicorn", "starlette"):
    logging.getLogger(logger_name).addFilter(CancelledErrorFilter())


mcp = FastMCP(name="RoamResearch", stateless_http=True)
logger = logging.getLogger(__name__)

# Background task management
background_tasks: set[asyncio.Task] = set()


def create_background_task(coro):
    """Create a background task and track it for graceful shutdown."""
    task = asyncio.create_task(coro)
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    return task


async def shutdown_background_tasks(timeout=30):
    """Wait for all background tasks to complete with timeout."""
    if background_tasks:
        logger.info(f"Waiting for {len(background_tasks)} background tasks to complete...")
        try:
            await asyncio.wait_for(
                asyncio.gather(*background_tasks, return_exceptions=True),
                timeout=timeout
            )
            logger.info("All background tasks completed successfully")
        except asyncio.TimeoutError:
            logger.warning(f"Background tasks timeout after {timeout}s, some tasks may be incomplete")
            # Cancel remaining tasks
            for task in background_tasks:
                task.cancel()


# Database management
def init_db():
    """Initialize SQLite database for task tracking."""
    db_path = project_root / 'storage' / 'tasks.db'
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute('PRAGMA journal_mode=WAL')  # Write-Ahead Logging for crash safety
    conn.execute('PRAGMA synchronous=NORMAL')

    conn.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            page_uid TEXT NOT NULL,
            title TEXT NOT NULL,
            markdown TEXT NOT NULL,
            status TEXT NOT NULL,
            total_blocks INTEGER,
            processed_blocks INTEGER DEFAULT 0,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def get_db_connection():
    """Get a database connection."""
    db_path = project_root / 'storage' / 'tasks.db'
    return sqlite3.connect(str(db_path))


def save_task(task_id: str, page_uid: str, title: str, markdown: str, status: str, total_blocks: int = 0):
    """Save a new task to database."""
    conn = get_db_connection()
    conn.execute('''
        INSERT INTO tasks (task_id, page_uid, title, markdown, status, total_blocks, processed_blocks)
        VALUES (?, ?, ?, ?, ?, ?, 0)
    ''', (task_id, page_uid, title, markdown, status, total_blocks))
    conn.commit()
    conn.close()


def update_task(task_id: str, status: str = None, processed_blocks: int = None, error_message: str = None):
    """Update task status and progress."""
    conn = get_db_connection()
    updates = []
    params = []

    if status is not None:
        updates.append('status = ?')
        params.append(status)
    if processed_blocks is not None:
        updates.append('processed_blocks = ?')
        params.append(processed_blocks)
    if error_message is not None:
        updates.append('error_message = ?')
        params.append(error_message)

    updates.append('updated_at = ?')
    params.append(datetime.now().isoformat())

    if status in ('completed', 'failed'):
        updates.append('completed_at = ?')
        params.append(datetime.now().isoformat())

    params.append(task_id)

    conn.execute(f'''
        UPDATE tasks
        SET {', '.join(updates)}
        WHERE task_id = ?
    ''', params)
    conn.commit()
    conn.close()


def get_task(task_id: str):
    """Get task information."""
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.execute('SELECT * FROM tasks WHERE task_id = ?', (task_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_when(when = None):
    if not when:
        date_obj = pendulum.now()
    else:
        try:
            date_obj = cast(pendulum.DateTime, pendulum.parse(when.strip()))
        except Exception as e:
            raise ValueError(f"Unrecognized date format: {when}")
    return date_obj


async def _process_content_blocks_background(task_id: str, page_uid: str, actions: list):
    """Process content blocks in batches in the background."""
    batch_size = int(os.getenv('BATCH_SIZE', '100'))
    max_retries = int(os.getenv('MAX_RETRIES', '3'))

    total_blocks = len(actions)
    processed = 0

    logger.info(f"Task {task_id}: Processing {total_blocks} blocks in batches of {batch_size}")
    update_task(task_id, status='processing')

    try:
        async with RoamClient() as client:
            # Process in batches
            for i in range(0, total_blocks, batch_size):
                batch = actions[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (total_blocks + batch_size - 1) // batch_size

                logger.info(f"Task {task_id}: Processing batch {batch_num}/{total_batches} ({len(batch)} blocks)")

                # Retry logic for this batch
                retry_count = 0
                last_error = None

                while retry_count <= max_retries:
                    try:
                        await client.batch_actions(batch)
                        processed += len(batch)
                        update_task(task_id, processed_blocks=processed)
                        logger.info(f"Task {task_id}: Batch {batch_num}/{total_batches} completed. Progress: {processed}/{total_blocks}")
                        break  # Success, move to next batch
                    except Exception as e:
                        retry_count += 1
                        last_error = str(e)
                        if retry_count <= max_retries:
                            wait_time = 2 ** retry_count  # Exponential backoff: 2s, 4s, 8s
                            logger.warning(f"Task {task_id}: Batch {batch_num} failed (attempt {retry_count}/{max_retries}), retrying in {wait_time}s: {e}")
                            await asyncio.sleep(wait_time)
                        else:
                            # Max retries exceeded
                            error_msg = f"Batch {batch_num} failed after {max_retries} retries: {last_error}"
                            logger.error(f"Task {task_id}: {error_msg}")
                            update_task(task_id, status='failed', error_message=error_msg, processed_blocks=processed)
                            return

        # All batches completed successfully
        update_task(task_id, status='completed', processed_blocks=processed)
        logger.info(f"Task {task_id}: All {total_blocks} blocks processed successfully")

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"Task {task_id}: {error_msg}")
        update_task(task_id, status='failed', error_message=error_msg, processed_blocks=processed)


async def get_topic_uid(client, topic: str, when: pendulum.DateTime):
    block = await client.q(f"""
        [:find (pull ?id [:block/uid :node/title :block/string])
         :where [?id :block/string "{topic}"]
                [?id :block/parents ?pid]
                [?pid :block/uid "{when.format('MM-DD-YYYY')}"]
        ]
    """)
    if not block:
        raise ValueError(f"Topic node {topic} not found for {when.format('MM-DD-YYYY')}")
    return block[0][0][':block/uid']

#
#
#


@mcp.tool(
    description="""Save a markdown doc into Roam Research's Daily Notes.

    - title: should be plaintext of the document title.
    - markdown: should be GitHub Flavored Markdown (GFM) format. Do not include title as H1 in markdown.
    """
)
async def save_markdown(title: str, markdown: str) -> str:
    # Generate unique IDs
    task_id = uuid.uuid4().hex
    link_block_uid = uuid.uuid4().hex  # UID for the link block (this is the deterministic return value)

    try:
        # Create page and generate content actions
        page = create_page(title)
        page_uid = page['page']['uid']
        content_actions = gfm_to_batch_actions(markdown, page_uid)

        logger.info(f"Task {task_id}: Page UID: {page_uid}, Content blocks: {len(content_actions)}")

        # Save task to database
        save_task(task_id, page_uid, title, markdown, 'pending', len(content_actions))

        # Phase 1 (Synchronous): Create page + link block
        when = get_when()
        topic_node = os.getenv("TOPIC_NODE")

        async with RoamClient() as client:
            if topic_node:
                topic_uid = await get_topic_uid(client, topic_node, when)
                logger.info(f"Task {task_id}: Topic UID: {topic_uid}")
                link_action = create_block(f"[[{title}]]", topic_uid, link_block_uid)
            else:
                link_action = create_block(f"[[{title}]]", when.format('MM-DD-YYYY'), link_block_uid)

            # Submit page + link block synchronously
            await client.batch_actions([page, link_action])
            logger.info(f"Task {task_id}: Page and link block created successfully")

        # Update task status
        update_task(task_id, status='link_created')

        # Phase 2 (Background): Process content blocks
        if content_actions:
            create_background_task(
                _process_content_blocks_background(task_id, page_uid, content_actions)
            )
            logger.info(f"Task {task_id}: Background processing started for {len(content_actions)} blocks")

        # Return immediately
        return f"Task {task_id} started. Page [[{title}]] created with link block {link_block_uid}. Processing {len(content_actions)} content blocks in background."

    except Exception as e:
        logger.error(f"Task {task_id}: Error during initial setup: {e}\n{traceback.format_exc()}")
        error_msg = f"Error: {e}"
        if type(e) == httpx.HTTPStatusError:
            error_msg = f"Error: {e.response.text}\n\n{e.response.status_code}"
        update_task(task_id, status='failed', error_message=error_msg)
        return error_msg
    finally:
        # Save debug file (always, for recovery purposes)
        storage_dir = os.getenv("ROAM_STORAGE_DIR")
        if storage_dir:
            try:
                directory = Path(storage_dir)
                directory.mkdir(parents=True, exist_ok=True)
                dt = pendulum.now().format('YYYYMMDD')
                debug_file = directory / f"{dt}_{link_block_uid}.md"
                with open(debug_file, 'w') as f:
                    f.write(f"{title}\n\n{markdown}")
                logger.info(f"Task {task_id}: Debug file saved: {debug_file}")
            except Exception as storage_error:
                logger.warning(f"Task {task_id}: Failed to write debug file: {storage_error}")
        else:
            logger.info(f"Task {task_id}: ROAM_STORAGE_DIR not set; skipped saving debug file.")


@mcp.tool(name="query", description="Query your Roam Research data with datalog, query MUST be a valid datalog query")
async def handle_query_roam(q: str) -> str:
    async with RoamClient() as client:
        try:
            result = await client.q(q)
            if result:
                return result
            return pprint.pformat(result)
        except httpx.HTTPStatusError as e:
            print(e.response.text)
            return f"Error: {e}"
        except Exception as e:
            return f"{type(e)}: {e}"


@mcp.tool(
    description="""Get the journaling for a given date. If no date is provided, the current date will be used.
    The date string should be in ISO8601, RFC2822, or RFC3339 format.
    Example:
    ```
    get_journaling_by_date("2021-01-01")
    get_journaling_by_date("2021-01-01T00:00:00Z")
    get_journaling_by_date("2021-01-01 00:00:00")
    ```
    """
)
async def get_journaling_by_date(when=None):
    if not when:
        date_obj = pendulum.now()
    else:
        try:
            date_obj = cast(pendulum.DateTime, pendulum.parse(when.strip()))
        except Exception as e:
            return f"Unrecognized date format: {when}"
    logger.info('get_journaling_by_date: %s', date_obj)
    topic_node = os.getenv("TOPIC_NODE")
    logger.info('topic_node: %s', topic_node)
    if topic_node:
        query = f"""
[:find (pull ?e [*
                 {{:block/children [*]}}
                 {{:block/refs [*]}}
                ])
 :where
    [?id :block/string "{topic_node}"]
    [?id :block/parents ?pid]
    [?pid :block/uid "{date_obj.format('MM-DD-YYYY')}"]
    [?e :block/parents ?id]
    [?e :block/parents ?pid]
]
        """
    else:
        query = f"""
[:find (pull ?e [*
                 {{:block/children [*]}}
                 {{:block/refs [*]}}
                ])
 :where
    [?pid :block/uid "{date_obj.format('MM-DD-YYYY')}"]
    [?e :block/parents ?id]
    [?e :block/parents ?pid]
]
        """
    async with RoamClient() as client:
        resp = await client.q(query)
    if resp is None:
        return ''
    logger.info(f'get_journaling_by_date: found {len(resp)} blocks')
    nodes = list(chain(*(i for i in resp)))
    if topic_node:
        root = list(sorted([i for i in nodes if len(i.get(':block/parents', [])) == 2], key=lambda i: i[':block/order']))
    else:
        root = list(sorted([i for i in nodes if len(i.get(':block/parents', [])) == 1], key=lambda i: i[':block/order']))
    blocks = []
    for i in root:
        blocks.append(format_block(i, nodes))
    if not blocks:
        return ''
    return "\n\n".join(blocks).strip()


#
#
#


async def serve(host: str | None = None, port: int | None = None):
    load_dotenv()

    # Initialize database
    init_db()
    logger.info("Database initialized")

    import uvicorn

    app = mcp.streamable_http_app()
    # FIXME: mount for SSE endpoint, but missed the authorization middleware.
    app.routes.extend(mcp.sse_app().routes)

    env_host = os.getenv("HOST")
    env_port = os.getenv("PORT")
    host = host or (str(env_host) if env_host else mcp.settings.host)
    default_port = 9000
    port = port or (int(env_port) if env_port else default_port)

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=mcp.settings.log_level.lower(),
    )
    server = uvicorn.Server(config)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("Received exit signal, shutting down...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass

    # Also handle SIGHUP if possible
    try:
        loop.add_signal_handler(signal.SIGHUP, _signal_handler)
    except (NotImplementedError, AttributeError):
        # SIGHUP may not be available on all platforms
        pass

    try:
        server_task = asyncio.create_task(server.serve())
        stop_task = asyncio.create_task(stop_event.wait())
        done, pending = await asyncio.wait(
            [server_task, stop_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        if stop_task in done:
            await server.shutdown()
        await server_task
    except KeyboardInterrupt:
        logger.info("Server interrupted by KeyboardInterrupt")
    except asyncio.CancelledError:
        logger.info("Server cancelled and exited gracefully.")
    finally:
        # Wait for background tasks to complete
        await shutdown_background_tasks()
        logger.info("All resources cleaned up, exiting.")


if __name__ == "__main__":
    asyncio.run(serve())
