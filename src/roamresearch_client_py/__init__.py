#!/usr/bin/env python

import os
import re
import json
import uuid
from httpx import AsyncClient


ROAM_API_GRAPH = os.environ.get('ROAM_API_GRAPH')
ROAM_API_TOKEN = os.environ.get('ROAM_API_TOKEN')


"""
[:find
 (pull ?id [:block/uid :create/time :node/title :block/string])
 :in $ ?s-ts ?e-ts
 :where [?id :create/time ?t]
        [(< ?s-ts ?t)]
        [(< ?t ?e-ts)]
        (or [?id :block/string _] [?id :node/title _])
 ]
"""

#
# Foudation request functions
#

async def send_roam_request(client, path, data):
    url = f"https://api.roamresearch.com/api/graph/{ROAM_API_GRAPH}{path}"
    headers = {
        'Content-Type': 'application/json; charset=utf-8',
        'X-Authorization': f"Bearer {ROAM_API_TOKEN}"
    }
    data = json.dumps(data)
    resp = await client.post(url, headers=headers, data=data, follow_redirects=True)
    try:
        return resp.json()
    except Exception as err:
        pass

#
# Operations
#

async def roam_create_block(client, text, parent_uid):
    uid = uuid.uuid4().hex
    await send_roam_request(client, '/write', {
        "action": "create-block",
        "location": {
            "parent-uid": parent_uid,
            "order": "last",
        },
        "block": {
            "uid": uid,
            "string": text,
        },
    })
    return uid

async def roam_q(client, query, args=None):
    if args is None:
        args = []
    body = await send_roam_request(client, '/q', {
        "query": query,
        "args": args
    })
    return body

async def roam_batch_actions(client, actions):
    await send_roam_request(client, '/write', {
        "action": "batch-actions",
        "actions": actions
    })


#
# actions
#

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
    """ Experimental class to create blocks in a more fluent way.

        Example:

        ```
        b = Block('first line', parent_uid=resp['result'][0][0][':block/uid'])
        with b as p:
            p.append('hi')
            p.append('what you done yesterday?')
            with p:
                p.append('lv2')
            p.append('what next?')
            p.append('happy with me?')
        ```
    """
    def __init__(self, text: str, parent_uid=None, open=True):
        self.actions = []
        # Create the root block.
        uid = uuid.uuid4().hex
        self.actions.append(create_block(text, parent_uid=parent_uid, uid=uid, open=open))
        # Remeber the initial context.
        self.current_parent_uid = parent_uid
        self.current_uid = uid
        self.parent_uid_stack = []

    def append(self, text: str):
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

    async def save(self, client: AsyncClient):
        await send_roam_request(client, '/write', {
            'action': 'batch-actions',
            'actions': self.actions
        })


class Page:
    def __init__(self, title: str):
        self.page_uid = uuid.uuid4().hex
        self.page_title = title
        self.blocks = []

    def add(self, text: str):
        block = Block(text, parent_uid=self.page_uid)
        self.blocks.append(block)
        return block

    def to_actions(self, append_to=None):
        actions = []
        actions.append(create_page(self.page_title, uid=self.page_uid))
        for block in self.blocks:
            actions.extend(block.actions)
        if append_to is not None:
            actions.append(create_block(
                f"[[{self.page_title}]]",
                parent_uid=append_to
            ))
        return actions

    async def save(self, client: AsyncClient, append_to=None):
        actions = self.to_actions(append_to=append_to)
        await send_roam_request(client, '/write', {
            'action': 'batch-actions',
            'actions': actions
        })