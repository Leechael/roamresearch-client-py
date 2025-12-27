import pytest


from roamresearch_client_py.diff import ExistingBlock, diff_block_trees, generate_batch_actions
from roamresearch_client_py.structs import Block, BlockRef


def _idx(actions: list[dict], predicate) -> int:
    for i, action in enumerate(actions):
        if predicate(action):
            return i
    raise AssertionError("action not found")


def test_diff_moves_child_out_before_deleting_parent():
    """
    Regression: if a child block is preserved but its parent is removed,
    we must move the child out before deleting the parent (delete is recursive).
    """
    page_uid = "page"

    parent = ExistingBlock(uid="A", text="Parent", order=0, heading=None, parent_uid=page_uid)
    child = ExistingBlock(uid="B", text="Keep", order=0, heading=None, parent_uid="A")
    parent.children = [child]

    new_child = Block("Keep", parent_ref=page_uid, ref=BlockRef(block_uid="newB"))

    diff = diff_block_trees([parent], [new_child], page_uid)
    actions = generate_batch_actions(diff)

    move_i = _idx(
        actions,
        lambda a: a["action"] == "move-block"
        and a["block"]["uid"] == "B"
        and a["location"]["parent-uid"] == page_uid,
    )
    delete_i = _idx(
        actions,
        lambda a: a["action"] == "delete-block" and a["block"]["uid"] == "A",
    )
    assert move_i < delete_i


def test_create_under_matched_parent_uses_existing_parent_uid():
    """
    Regression: when a parent block is matched to an existing UID, any newly-created
    children must target the existing parent UID (not the new temporary UID).
    """
    page_uid = "page"

    existing_parent = ExistingBlock(uid="P", text="Parent", order=0, heading=None, parent_uid=page_uid)

    new_parent = Block("Parent", parent_ref=page_uid, ref=BlockRef(block_uid="newP"))
    new_child = Block("Child", parent_ref=BlockRef(block_uid="newP"), ref=BlockRef(block_uid="newC"))

    diff = diff_block_trees([existing_parent], [new_parent, new_child], page_uid)

    create_child = next(a for a in diff.creates if a["block"]["uid"] == "newC")
    assert create_child["location"]["parent-uid"] == "P"

