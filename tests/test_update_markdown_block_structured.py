from roamresearch_client_py.gfm_to_roam import gfm_to_blocks
import roamresearch_client_py.server as server
import pytest

from roamresearch_client_py.server import (
    BLOCK_UPDATE_MARKDOWN_NEWLINE_THRESHOLD,
    _should_parse_block_update_as_markdown,
    _split_root_and_children_markdown,
)


def test_should_parse_block_update_as_markdown_true_for_many_newlines():
    text = "line\n" * (BLOCK_UPDATE_MARKDOWN_NEWLINE_THRESHOLD + 2)
    assert _should_parse_block_update_as_markdown(text)


def test_should_parse_block_update_as_markdown_true_for_heading_prefix():
    assert _should_parse_block_update_as_markdown("# Title\n- a")
    assert _should_parse_block_update_as_markdown("## Title")


def test_should_parse_block_update_as_markdown_false_for_short_text():
    assert not _should_parse_block_update_as_markdown("just a line")
    assert not _should_parse_block_update_as_markdown("")


def test_split_root_and_children_markdown_heading_root():
    root_text, root_heading, children_md = _split_root_and_children_markdown("# Title\n- a\n- b\n")
    assert root_text == "Title"
    assert root_heading == 1
    assert children_md.strip() == "- a\n- b"


def test_split_root_and_children_markdown_non_heading_root():
    root_text, root_heading, children_md = _split_root_and_children_markdown("Intro line\n- a\n")
    assert root_text == "Intro line"
    assert root_heading is None
    assert children_md.strip() == "- a"


def test_split_root_and_children_markdown_heading_clamps_to_3():
    root_text, root_heading, children_md = _split_root_and_children_markdown("#### Too Deep\n- a\n")
    assert root_text == "Too Deep"
    assert root_heading == 3
    assert children_md.strip() == "- a"


def test_gfm_to_blocks_skip_h1_default_true_ignores_h1():
    page_uid = "page"
    blocks = gfm_to_blocks("# H1\n- a\n", page_uid)
    assert blocks
    assert blocks[0].text != "H1"


def test_gfm_to_blocks_skip_h1_false_keeps_h1_and_nests_children():
    page_uid = "page"
    blocks = gfm_to_blocks("# H1\n- a\n", page_uid, skip_h1=False)
    assert len(blocks) >= 2
    assert blocks[0].text == "H1"
    assert blocks[0].heading == 1
    assert blocks[1].parent_ref == blocks[0].ref


def test_gfm_to_blocks_converts_task_markers_to_roam_todo_done():
    page_uid = "page"
    blocks = gfm_to_blocks("- [ ] a\n- [x] b\n- [X] c\n", page_uid)
    assert [b.text for b in blocks] == ["{{[[TODO]]}} a", "{{[[DONE]]}} b", "{{[[DONE]]}} c"]


def test_gfm_to_blocks_converts_task_markers_in_ordered_list():
    page_uid = "page"
    blocks = gfm_to_blocks("1. [ ] a\n2. [x] b\n", page_uid)
    assert [b.text for b in blocks] == ["1. {{[[TODO]]}} a", "2. {{[[DONE]]}} b"]


class _FakeRoamClient:
    def __init__(self, *, block_by_uid: dict | None = None):
        self._block_by_uid = block_by_uid
        self.batch_actions_calls: list[list[dict]] = []
        self.update_block_text_calls: list[tuple[str, str, bool]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get_block_by_uid(self, uid: str):
        return self._block_by_uid

    async def update_block_text(self, uid: str, text: str, *, dry_run: bool = False):
        self.update_block_text_calls.append((uid, text, dry_run))
        return {"actions": [], "stats": {"updates": 1, "creates": 0, "moves": 0, "deletes": 0}}

    async def batch_actions(self, actions: list[dict]):
        self.batch_actions_calls.append(actions)
        return {}


@pytest.mark.anyio
async def test_update_markdown_block_plain_text_uses_update_block_text(monkeypatch):
    fake = _FakeRoamClient()
    monkeypatch.setattr(server, "RoamClient", lambda: fake)

    # Not multi-line and not a heading -> should not be parsed as markdown.
    result = await server.update_markdown("abc123", "just a line", False)
    assert "Updated:" in result
    assert fake.update_block_text_calls == [("abc123", "just a line", False)]
    assert fake.batch_actions_calls == []


@pytest.mark.anyio
async def test_update_markdown_block_plain_text_converts_task_marker(monkeypatch):
    fake = _FakeRoamClient()
    monkeypatch.setattr(server, "RoamClient", lambda: fake)

    result = await server.update_markdown("abc123", "[ ] task", False)
    assert "Updated:" in result
    assert fake.update_block_text_calls == [("abc123", "{{[[TODO]]}} task", False)]
    assert fake.batch_actions_calls == []


@pytest.mark.anyio
async def test_update_markdown_block_structured_heading_updates_root_and_children(monkeypatch):
    existing = {
        ":block/uid": "u1",
        ":block/string": "Old",
        ":block/heading": None,
        ":block/order": 0,
        ":block/children": [],
    }
    fake = _FakeRoamClient(block_by_uid=existing)
    monkeypatch.setattr(server, "RoamClient", lambda: fake)

    markdown = "# Title\n- a\n- b\n"
    result = await server.update_markdown("u1", markdown, False)
    assert result.startswith("Updated:")
    assert fake.update_block_text_calls == []
    assert len(fake.batch_actions_calls) == 1

    actions = fake.batch_actions_calls[0]
    assert actions[0]["action"] == "update-block"
    assert actions[0]["block"]["uid"] == "u1"
    assert actions[0]["block"]["string"] == "Title"
    assert actions[0]["block"]["heading"] == 1

    create_actions = [a for a in actions if a["action"] == "create-block"]
    assert {a["block"]["string"] for a in create_actions} == {"a", "b"}
    assert all(a["location"]["parent-uid"] == "u1" for a in create_actions)


@pytest.mark.anyio
async def test_update_markdown_block_structured_dry_run_does_not_call_batch_actions(monkeypatch):
    existing = {
        ":block/uid": "u1",
        ":block/string": "Old",
        ":block/heading": None,
        ":block/order": 0,
        ":block/children": [],
    }
    fake = _FakeRoamClient(block_by_uid=existing)
    monkeypatch.setattr(server, "RoamClient", lambda: fake)

    result = await server.update_markdown("u1", "# Title\n- a\n- b\n", True)
    assert result.startswith("Dry run - would make:")
    assert "2 creates" in result
    assert "1 updates" in result
    assert fake.batch_actions_calls == []


@pytest.mark.anyio
async def test_update_markdown_block_structured_non_heading_clears_existing_heading(monkeypatch):
    existing = {
        ":block/uid": "u1",
        ":block/string": "Old",
        ":block/heading": 2,
        ":block/order": 0,
        ":block/children": [],
    }
    fake = _FakeRoamClient(block_by_uid=existing)
    monkeypatch.setattr(server, "RoamClient", lambda: fake)

    # Many newlines triggers structured mode even without a heading on the first line.
    # Use a thematic break for the remaining content so it parses into no child blocks.
    markdown = "New root\n\n\n\n\n\n---"
    result = await server.update_markdown("u1", markdown, False)
    assert result.startswith("Updated:")
    assert len(fake.batch_actions_calls) == 1
    actions = fake.batch_actions_calls[0]
    assert actions == [{"action": "update-block", "block": {"uid": "u1", "string": "New root", "heading": 0}}]
