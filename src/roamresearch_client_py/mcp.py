from typing import List, cast
import pprint
from itertools import chain
import logging
import signal
import os
import asyncio
import uuid

from dotenv import load_dotenv
import mcp.types as types
from mcp.server.fastmcp import FastMCP
import httpx
import pendulum

from .RoamClient import RoamClient, create_page, create_block
from .formatter import format_block
from .gfm_to_roam import gfm_to_batch_actions


class CancelledErrorFilter(logging.Filter):
    def filter(self, record):
        return "asyncio.exceptions.CancelledError" not in record.getMessage()

for logger_name in ("uvicorn.error", "uvicorn.access", "uvicorn", "starlette"):
    logging.getLogger(logger_name).addFilter(CancelledErrorFilter())


mcp = FastMCP(name="RoamResearch", stateless_http=True)
logger = logging.getLogger(__name__)


def get_when(when = None):
    if not when:
        date_obj = pendulum.now()
    else:
        try:
            date_obj = cast(pendulum.DateTime, pendulum.parse(when.strip()))
        except Exception as e:
            raise ValueError(f"Unrecognized date format: {when}")
    return date_obj


async def get_topic_uid(client, topic: str, when: pendulum.DateTime):
    print('get_topic_uid', topic, when)
    block = await client.q(f"""
        [:find (pull ?id [:block/uid :node/title :block/string])
         :where [?id :block/string "{topic}"]
                [?id :block/parents ?pid]
                [?pid :block/uid "{when.format('MM-DD-YYYY')}"]
        ]
    """)
    print(block)
    if not block or not block['result']:
        raise ValueError(f"Topic node {topic} not found for {when.format('MM-DD-YYYY')}")
    return block['result'][0][0][':block/uid']

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
    async with RoamClient() as client:
        page = create_page(title)
        actions = gfm_to_batch_actions(markdown, page['page']['uid'])
        actions = [page] + actions
        when = get_when()
        topic_node = os.getenv("TOPIC_NODE")
        print('topic_node', topic_node)
        if topic_node:
            topic_uid = await get_topic_uid(client, topic_node, when)
            actions.append(create_block(f"[[{title}]]", topic_uid, uuid.uuid4().hex))
        else:
            actions.append(create_block(f"[[{title}]]", when.format('MM-DD-YYYY'), uuid.uuid4().hex))
        await client.batch_actions(actions)
    return f"Saved"


@mcp.tool(name="query", description="Query your Roam Research data with datalog, query MUST be a valid datalog query")
async def handle_query_roam(q: str) -> str:
    print(q)
    async with RoamClient() as client:
        try:
            result = await client.q(q)
            if result and result.get("result"):
                return result["result"]
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
    logger.info(f'get_journaling_by_date: found {len(resp["result"])} blocks')
    nodes = list(chain(*(i for i in resp['result'])))
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


async def serve():
    load_dotenv()

    import uvicorn

    app = mcp.streamable_http_app()
    # FIXME: mount for SSE endpoint, but missed the authorization middleware.
    app.routes.extend(mcp.sse_app().routes)

    host = os.getenv("HOST") and str(os.getenv("HOST")) or mcp.settings.host
    port = os.getenv("PORT") and int(os.getenv("PORT", 8000)) or mcp.settings.port

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
        logger.info("All resources cleaned up, exiting.")


if __name__ == "__main__":
    asyncio.run(serve())
