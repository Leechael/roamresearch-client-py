from typing import cast, Optional, Union
import json
import uuid
import os

import httpx
import pendulum

def create_page(title, uid=None, childrenViewType=None):
    if childrenViewType is None:
        childrenViewType = "bullet"
    return {
        "action": "create-page",
        "page": {
            "title": title,
            "uid": uid,
            "children-view-type": childrenViewType,
        },
    }


def create_block(text, parent_uid, uid=None, order="last", open=True):
    return {
        "action": "create-block",
        "location": {
            "parent-uid": parent_uid,
            "order": order,
        },
        "block": {
            "uid": uid,
            "string": text,
            "open": open,
        },
    }


def update_block(uid, text):
    return {
        "action": "update-block",
        "block": {
            "uid": uid,
            "string": text,
        },
    }


def remove_block(uid):
    return {
        "action": "delete-block",
        "block": {
            "uid": uid,
        },
    }


class Block:
    def __init__(
            self,
            text: str,
            parent_uid=None,
            open=True,
            client: Union["RoamClient", None] = None
    ):
        self.client = client
        self.actions = []
        # Create the root block.
        uid = uuid.uuid4().hex
        self.actions.append(create_block(text, parent_uid=parent_uid, uid=uid, open=open))
        # Remeber the initial context.
        self.current_parent_uid = parent_uid
        self.current_uid = uid
        self.parent_uid_stack = []

    def set_client(self, client: "RoamClient"):
        self.client = client

    def text(self, text: str):
        for i in self.actions:
            if i['block']['uid'] == self.current_uid:
                i['block']['string'] = text
                break

    def append_text(self, text: str):
        for i in self.actions:
            if i['block']['uid'] == self.current_uid:
                i['block']['string'] += text
                break

    def write(self, text: str):
        uid = uuid.uuid4().hex
        self.actions.append(create_block(text, parent_uid=self.current_parent_uid, uid=uid))
        self.current_uid = uid
        return self

    def __enter__(self):
        self.parent_uid_stack.append(self.current_parent_uid)
        self.current_parent_uid = self.current_uid
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.current_parent_uid = self.parent_uid_stack.pop()
        return False

    def to_actions(self):
        return [i for i in self.actions]

    async def save(self):
        assert self.client is not None, "Client not initialized"
        await self.client.batch_actions(self.actions)
  
    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self.actions:
            await self.save()

    def __str__(self):
        return json.dumps(self.actions)


class RoamClient(object):
    def __init__(self, api_token: str | None = None, graph: str | None = None):
        if api_token is None:
            api_token = os.getenv("ROAM_API_TOKEN")
        if graph is None:
            graph = os.getenv("ROAM_API_GRAPH")
        if api_token is None or graph is None:
            raise Exception("ROAM_API_TOKEN and ROAM_API_GRAPH must be set")
        self.api_token = api_token
        self.graph = graph
        self._client = None

    def connect(self):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-Authorization': f"Bearer {self.api_token}"
        }
        self._client = httpx.AsyncClient(
            base_url=f"https://api.roamresearch.com/api/graph/{self.graph}",
            headers=headers,
            follow_redirects=True,
            timeout=10.0,
        )
        return self

    async def disconnect(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        self.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self._client is not None:
            await self._client.aclose()

    async def _request(self, path, data, parse_response=True):
        assert self._client is not None, "Client not initialized"
        resp = await self._client.post(path, data=json.dumps(data))  # type: ignore
        resp.raise_for_status()
        if parse_response:
            return resp.json()

    async def q(self, query: str, args: Optional[list[str]] = None):
        return await self._request("/q", {
            "query": query,
            "args": args or [],
        })

    async def batch_actions(self, actions: list[dict]):
        return await self._request("/write", {
            "action": "batch-actions",
            "actions": actions,
        })

    async def write(self, text: str, parent_uid: str | None = None, uid: str | None = None, order: str = "last", open: bool = True):
        if parent_uid is None:
            now = pendulum.now()
            parent_uid = f"{now.month:02d}-{now.day:02d}-{now.year}"
        if not uid:
            uid = uuid.uuid4().hex
        data = create_block(text, parent_uid, uid, order, open)
        return await self._request("/write", data, parse_response=False)

    async def get_daily_page(self, date: pendulum.Date | None = None):
        if date is None:
            date = pendulum.now().date()
        # Step 1: ensure the daily page has been created.
        resp = await self.q(
            f"[:find (pull ?id [:block/uid :node/title]) :where [?id :block/uid \"{date.format('MM-DD-YYYY')}\"]]"
        )
        if not resp or not resp.get('result'):
            # TODO create the daily page first.
            raise Exception('Daily page not found.')
        return resp.get('result')[0][0]

    async def get_block_recursively(self, uid: str):
        query = f"""
[:find (pull ?e [*
                 {{:block/children [*]}}
                 {{:block/refs [*]}}
                ])
 :where
    [?pid :block/uid "{uid}"]
    [?e :block/parents ?id]
    [?e :block/parents ?pid]
]
        """
        resp = await self.q(query)
        return resp

    def create_block(self, text: str, parent_uid: str | None = None, open: bool = True):
        if parent_uid is None:
            now = pendulum.now()
            parent_uid = f"{now.month:02d}-{now.day:02d}-{now.year}"
        return Block(text, parent_uid, open, client=self)