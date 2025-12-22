from typing import cast, Optional, Union
import json
import uuid
import os

import httpx
import pendulum


class RoamRequestError(ValueError):
    pass


def create_page(title, uid=None, childrenViewType=None):
    if childrenViewType is None:
        childrenViewType = "bullet"
    if uid:
        return {
            "action": "create-page",
            "page": {
                "title": title,
                "uid": uid,
                "children-view-type": childrenViewType,
            },
        }
    else:
        return {
            "action": "create-page",
            "page": {
                "title": title,
                "uid": uuid.uuid4().hex,
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


def move_block(uid: str, parent_uid: str, order: int | str = "last"):
    """Generate move-block action for batch operations."""
    return {
        "action": "move-block",
        "block": {"uid": uid},
        "location": {
            "parent-uid": parent_uid,
            "order": order
        }
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
    _logging_configured = False

    def __init__(self, api_token: str | None = None, graph: str | None = None):
        from .config import get_env_or_config, configure_logging

        # Configure logging once on first client instantiation
        if not RoamClient._logging_configured:
            configure_logging()
            RoamClient._logging_configured = True

        if api_token is None:
            api_token = get_env_or_config("ROAM_API_TOKEN", "roam.api_token")
        if graph is None:
            graph = get_env_or_config("ROAM_API_GRAPH", "roam.api_graph")
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
        if resp.status_code == 500 or resp.status_code == 400:
            # NOTE: not sure is it always return JSON.
            payload = resp.json()
            if "message" in payload:
                raise RoamRequestError(payload["message"])
            else:
                raise RoamRequestError(resp.text)
        resp.raise_for_status()
        if parse_response:
            return resp.json()

    async def q(self, query: str, args: Optional[list[str]] = None):
        resp = await self._request("/q", {
            "query": query,
            "args": args or [],
        })
        assert resp is not None
        return resp['result']

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

    async def get_block_by_db_id(self, db_id: int):
        query = f"""
[:find (pull ?e [*])
 :where
 [?e _ _]
 [(= ?e {db_id})]]
        """
        resp = await self.q(query)
        return resp

    def create_block(self, text: str, parent_uid: str | None = None, open: bool = True):
        if parent_uid is None:
            now = pendulum.now()
            parent_uid = f"{now.month:02d}-{now.day:02d}-{now.year}"
        return Block(text, parent_uid, open, client=self)

    async def get_page_by_title(self, title: str):
        """Get a page and all its children by title."""
        # Escape quotes in title
        escaped_title = title.replace('"', '\\"')
        query = f'''
[:find (pull ?e [*
                 {{:block/children ...}}
                 {{:block/refs [*]}}
                ])
 :where [?e :node/title "{escaped_title}"]]
'''
        resp = await self.q(query)
        return resp[0][0] if resp else None

    async def search_blocks(
        self,
        terms: list[str],
        limit: int = 20,
        case_sensitive: bool = True,
        page_title: str | None = None
    ):
        """
        Search blocks containing all given terms.

        Args:
            terms: List of search terms (all must match)
            limit: Maximum number of results
            case_sensitive: Whether to match case
            page_title: Optional page title to scope the search

        Returns:
            List of [block_uid, block_string, page_title] tuples, sorted by relevance:
            - Priority 0: Page title exactly matches a search term
            - Priority 1: Block contains [[term]] or #[[term]]
            - Priority 2: Partial match
        """
        # Build conditions for each term
        conditions = []
        for term in terms:
            escaped = term.replace('"', '\\"')
            if case_sensitive:
                conditions.append(f'[(clojure.string/includes? ?s "{escaped}")]')
            else:
                conditions.append(f'[(clojure.string/includes? (clojure.string/lower-case ?s) "{escaped.lower()}")]')

        term_conditions = '\n    '.join(conditions)

        if page_title:
            # Search within specific page
            escaped_title = page_title.replace('"', '\\"')
            query = f'''
[:find ?uid ?s
 :where
    [?p :node/title "{escaped_title}"]
    [?b :block/page ?p]
    [?b :block/string ?s]
    [?b :block/uid ?uid]
    {term_conditions}]
'''
            resp = await self.q(query)
            # Add page_title to results
            results = [[uid, s, page_title] for uid, s in resp] if resp else []
        else:
            # Global search with page title
            query = f'''
[:find ?uid ?s ?page-title
 :where
    [?b :block/string ?s]
    [?b :block/uid ?uid]
    [?b :block/page ?p]
    [?p :node/title ?page-title]
    {term_conditions}]
'''
            resp = await self.q(query)
            results = list(resp) if resp else []

        # Sort by relevance
        def sort_key(item):
            uid, text, page = item[0], item[1], item[2]
            for term in terms:
                term_check = term if case_sensitive else term.lower()
                page_check = page if case_sensitive else page.lower()
                text_check = text if case_sensitive else text.lower()

                # Priority 0: Page title exactly matches
                if page_check == term_check:
                    return (0, page)

                # Priority 1: Contains [[term]] or #[[term]]
                if f'[[{term_check}]]' in text_check or f'#[[{term_check}]]' in text_check:
                    return (1, page)

            # Priority 2: Partial match
            return (2, page)

        results.sort(key=sort_key)
        return results[:limit]

    async def get_block_by_uid(self, uid: str):
        """Get a block and all its children by uid."""
        query = f'''
[:find (pull ?e [*
                 {{:block/children ...}}
                 {{:block/refs [*]}}
                ])
 :where [?e :block/uid "{uid}"]]
'''
        resp = await self.q(query)
        return resp[0][0] if resp else None

    async def update_block_text(
        self,
        uid: str,
        text: str,
        dry_run: bool = False
    ) -> dict:
        """
        Update a single block's text (does not process children).

        Args:
            uid: Block UID to update
            text: New text content
            dry_run: If True, return the action without executing

        Returns:
            dict with 'actions' and 'stats'

        Raises:
            ValueError: If block not found
        """
        # Verify block exists
        existing = await self.get_block_by_uid(uid)
        if not existing:
            raise ValueError(f"Block not found: {uid}")

        action = {
            "action": "update-block",
            "block": {
                "uid": uid,
                "string": text
            }
        }

        result = {
            'actions': [action],
            'stats': {'updates': 1, 'creates': 0, 'moves': 0, 'deletes': 0}
        }

        if not dry_run:
            await self.batch_actions([action])

        return result

    async def update_page_markdown(
        self,
        title: str,
        markdown: str,
        dry_run: bool = False
    ) -> dict:
        """
        Update an existing page with new markdown content using smart diff.

        This method:
        1. Fetches the existing page content
        2. Converts new markdown to blocks
        3. Computes minimal diff (preserving UIDs where possible)
        4. Executes the diff actions

        Args:
            title: Page title to update
            markdown: New GFM markdown content
            dry_run: If True, return actions without executing

        Returns:
            dict with:
                - 'actions': list of executed actions
                - 'stats': counts of creates, updates, moves, deletes
                - 'preserved_uids': list of UIDs that were reused

        Raises:
            ValueError: If page not found or markdown is empty
        """
        from .diff import parse_existing_blocks, diff_block_trees, generate_batch_actions
        from .gfm_to_roam import gfm_to_blocks

        if not markdown.strip():
            raise ValueError("Cannot update with empty content")

        # Fetch existing page
        page = await self.get_page_by_title(title)
        if not page:
            raise ValueError(f"Page not found: {title}")

        page_uid = page.get(':block/uid')
        if not page_uid:
            raise ValueError(f"Page has no UID: {title}")

        # Parse existing blocks
        existing_blocks = parse_existing_blocks(page)

        # Convert new markdown to blocks
        new_blocks = gfm_to_blocks(markdown, page_uid)

        # Compute diff
        diff = diff_block_trees(existing_blocks, new_blocks, page_uid)

        # Generate ordered actions
        actions = generate_batch_actions(diff)

        result = {
            'actions': actions,
            'stats': diff.stats(),
            'preserved_uids': list(diff.preserved_uids)
        }

        if not dry_run and actions:
            await self.batch_actions(actions)

        return result