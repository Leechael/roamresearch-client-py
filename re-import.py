import os
import sys
import asyncio
import pathlib
from typing import cast
import uuid

from dotenv import load_dotenv
import pendulum

from src.roamresearch_client_py.client import RoamClient, create_page, create_block
from src.roamresearch_client_py.gfm_to_roam import gfm_to_batch_actions

load_dotenv()

def get_when(when = None):
    if not when:
        date_obj = pendulum.now()
    else:
        try:
            date_obj = cast(pendulum.DateTime, pendulum.parse(when.strip()))
        except Exception:
            raise ValueError(f"Unrecognized date format: {when}")
    return date_obj


async def get_topic_uid(client, topic: str, when: pendulum.DateTime):
    block = await client.q(f"""
        [:find (pull ?id [:block/uid :node/title :block/string])
         :where [?id :block/string "{topic}"]
                [?id :block/parents ?pid]
                [?pid :block/uid "{when.format('MM-DD-YYYY')}"]
        ]
    """)
    if not block or not block['result']:
        raise ValueError(f"Topic node {topic} not found for {when.format('MM-DD-YYYY')}")
    return block['result'][0][0][':block/uid']


async def main():
    if len(sys.argv) < 2:
        print("Usage: pdm run re-import.py [file_path]")
        sys.exit(1)
    filepath = pathlib.Path(sys.argv[1])
    if not filepath.exists():
        print(f"File not exists: {sys.argv[1]}")
        sys.exit(1)
    with open(filepath) as fp:
        title = fp.readline().strip()
        markdown = fp.read().strip()
    async with RoamClient() as client:
        page_uid = uuid.uuid4().hex
        page = create_page(title)
        print(f"Page: {page}")
        actions = gfm_to_batch_actions(markdown, page['page']['uid'])
        print(f"Actions size: {len(actions)}")
        actions = [page] + actions
        when = get_when()
        topic_node = os.getenv("TOPIC_NODE")
        if topic_node:
            topic_uid = await get_topic_uid(client, topic_node, when)
            print(f"Topic UID: {topic_uid}")
            actions.append(create_block(f"[[{title}]]", topic_uid, page_uid))
        else:
            actions.append(create_block(f"[[{title}]]", when.format('MM-DD-YYYY'), page_uid))
        await client.batch_actions(actions)


if __name__ == '__main__':
    asyncio.run(main())
