from __future__ import annotations

from dataclasses import dataclass, field


from roamresearch_client_py.diff import ExistingBlock, diff_block_trees, generate_batch_actions
from roamresearch_client_py.structs import Block, BlockRef


def _nb(uid: str, text: str, parent_uid: str) -> Block:
    return Block(text, parent_ref=BlockRef(block_uid=parent_uid), ref=BlockRef(block_uid=uid))


def _eb(
    uid: str,
    text: str,
    order: int,
    parent_uid: str,
    children: list[ExistingBlock] | None = None,
) -> ExistingBlock:
    block = ExistingBlock(uid=uid, text=text, order=order, heading=None, parent_uid=parent_uid)
    if children:
        block.children = children
    return block


def _find_action(actions: list[dict], action: str, uid: str) -> dict:
    for a in actions:
        if a.get("action") == action and a.get("block", {}).get("uid") == uid:
            return a
    raise AssertionError(f"missing action={action} uid={uid}")


def _idx(actions: list[dict], action: str, uid: str) -> int:
    for i, a in enumerate(actions):
        if a.get("action") == action and a.get("block", {}).get("uid") == uid:
            return i
    raise AssertionError(f"missing action={action} uid={uid}")


def test_delete_ancestor_moves_grandchild_out_before_deletes():
    """
    Extreme: existing Keep is nested 3 levels deep, but new markdown keeps it at root.

    If we delete ancestors first, Roam will recursively delete the grandchild, so
    we must move it out before deleting.
    """
    page_uid = "page"
    c = _eb("C", "Keep", 0, "B")
    b = _eb("B", "Mid", 0, "A", children=[c])
    a = _eb("A", "Top", 0, page_uid, children=[b])

    new_keep = _nb("newC", "Keep", page_uid)

    diff = diff_block_trees([a], [new_keep], page_uid)
    actions = generate_batch_actions(diff)

    move_c = _find_action(actions, "move-block", "C")
    assert move_c["location"]["parent-uid"] == page_uid

    assert _idx(actions, "move-block", "C") < _idx(actions, "delete-block", "B")
    assert _idx(actions, "move-block", "C") < _idx(actions, "delete-block", "A")


def test_delete_parent_with_multiple_kept_children_moves_all_before_parent_delete():
    """
    Extreme: delete a parent block while keeping multiple children.

    We must move *all* preserved children out before deleting the parent.
    """
    page_uid = "page"
    b = _eb("B", "Keep1", 0, "A")
    c = _eb("C", "Keep2", 1, "A")
    d = _eb("D", "Drop", 2, "A")
    a = _eb("A", "Parent", 0, page_uid, children=[b, c, d])

    new_b = _nb("newB", "Keep1", page_uid)
    new_c = _nb("newC", "Keep2", page_uid)

    diff = diff_block_trees([a], [new_b, new_c], page_uid)
    actions = generate_batch_actions(diff)

    assert _idx(actions, "move-block", "B") < _idx(actions, "delete-block", "A")
    assert _idx(actions, "move-block", "C") < _idx(actions, "delete-block", "A")


def test_multi_level_parent_mapping_for_created_grandchild():
    """
    Extreme: parent and child are matched to existing UIDs, grandchild is new.

    The grandchild create must reference the *existing* UID of the matched parent.
    """
    page_uid = "page"

    existing_child = _eb("Q", "Child", 0, "P")
    existing_parent = _eb("P", "Parent", 0, page_uid, children=[existing_child])

    new_parent = _nb("newP", "Parent", page_uid)
    new_child = _nb("newQ", "Child", "newP")
    new_grand = _nb("newR", "GrandChild", "newQ")

    diff = diff_block_trees([existing_parent], [new_parent, new_child, new_grand], page_uid)

    create_grand = next(a for a in diff.creates if a["block"]["uid"] == "newR")
    assert create_grand["location"]["parent-uid"] == "Q"


def test_large_reorder_only_produces_moves():
    """
    Extreme: 200 siblings reorder (reverse order).

    Should result in 0 creates/deletes and a move for each block.
    """
    page_uid = "page"
    existing = [
        ExistingBlock(uid=f"U{i}", text=f"T{i}", order=i, heading=None, parent_uid=page_uid)
        for i in range(200)
    ]
    new_blocks = [_nb(f"n{i}", f"T{i}", page_uid) for i in reversed(range(200))]

    diff = diff_block_trees(existing, new_blocks, page_uid)

    assert diff.stats()["creates"] == 0
    assert diff.stats()["deletes"] == 0
    assert diff.stats()["updates"] == 0
    assert diff.stats()["moves"] == 200

    orders = [a["location"]["order"] for a in diff.moves]
    assert sorted(orders) == list(range(200))


def test_many_duplicate_texts_does_not_degenerate_to_full_recreate():
    """
    Extreme: 50 blocks share identical text.

    New markdown keeps 45 of them and introduces 5 new unique blocks.
    This should not degenerate into deleting/creating everything.
    """
    page_uid = "page"
    existing = [
        ExistingBlock(uid=f"U{i}", text="Same", order=i, heading=None, parent_uid=page_uid)
        for i in range(50)
    ]
    new_blocks = [_nb(f"n{i}", "Same", page_uid) for i in range(45)] + [
        _nb(f"new{i}", f"New{i}", page_uid) for i in range(5)
    ]

    diff = diff_block_trees(existing, new_blocks, page_uid)
    stats = diff.stats()

    assert stats["creates"] == 5
    assert stats["deletes"] == 5
    assert stats["moves"] == 0
    assert stats["preserved"] == 45


def test_cross_parent_duplicate_prefers_closer_order_candidate():
    """
    Extreme: identical text exists under different parents.

    This test forces deterministic matching by giving candidates very different
    orders, so the closest-order candidate should be preserved.
    """
    page_uid = "page"

    x1 = _eb("X1", "Same", 1, "PA")
    pa = _eb("PA", "ParentA", 0, page_uid, children=[x1])

    x2 = _eb("X2", "Same", 100, "PB")
    pb = _eb("PB", "ParentB", 1, page_uid, children=[x2])

    new_pa = _nb("newPA", "ParentA", page_uid)
    new_same = _nb("newSame", "Same", "newPA")

    diff = diff_block_trees([pa, pb], [new_pa, new_same], page_uid)

    # We expect the preserved UID for "Same" to be X1 (the closer order candidate).
    assert "X1" in diff.preserved_uids
    assert "X2" not in diff.preserved_uids


@dataclass
class _Node:
    uid: str
    parent_uid: str | None
    children: list[str] = field(default_factory=list)


def _apply_actions_with_recursive_delete(
    page_uid: str,
    existing_trees: list[ExistingBlock],
    actions: list[dict],
) -> dict[str, _Node]:
    """
    Minimal in-memory interpreter for Roam-like block operations.

    Purpose: assert generated action order never references a missing parent UID,
    and that delete is recursive.
    """
    nodes: dict[str, _Node] = {page_uid: _Node(uid=page_uid, parent_uid=None)}

    def add_existing(block: ExistingBlock):
        nodes[block.uid] = _Node(uid=block.uid, parent_uid=block.parent_uid)
        parent = block.parent_uid or page_uid
        nodes[parent].children.append(block.uid)
        for child in block.children:
            add_existing(child)

    for t in existing_trees:
        add_existing(t)

    def recursive_delete(uid: str):
        if uid not in nodes:
            return
        for child_uid in list(nodes[uid].children):
            recursive_delete(child_uid)
        parent_uid = nodes[uid].parent_uid
        if parent_uid and parent_uid in nodes:
            nodes[parent_uid].children = [c for c in nodes[parent_uid].children if c != uid]
        del nodes[uid]

    for a in actions:
        match a.get("action"):
            case "create-block":
                uid = a["block"]["uid"]
                parent_uid = a["location"]["parent-uid"]
                if parent_uid not in nodes:
                    raise AssertionError(f"create references missing parent {parent_uid}")
                if uid in nodes:
                    raise AssertionError(f"create duplicates uid {uid}")
                nodes[uid] = _Node(uid=uid, parent_uid=parent_uid)
                order = a["location"].get("order", "last")
                if order == "last":
                    nodes[parent_uid].children.append(uid)
                else:
                    nodes[parent_uid].children.insert(int(order), uid)

            case "move-block":
                uid = a["block"]["uid"]
                parent_uid = a["location"]["parent-uid"]
                if uid not in nodes:
                    raise AssertionError(f"move references missing uid {uid}")
                if parent_uid not in nodes:
                    raise AssertionError(f"move references missing parent {parent_uid}")
                old_parent = nodes[uid].parent_uid
                if old_parent and old_parent in nodes:
                    nodes[old_parent].children = [c for c in nodes[old_parent].children if c != uid]
                nodes[uid].parent_uid = parent_uid
                order = a["location"].get("order", "last")
                if order == "last":
                    nodes[parent_uid].children.append(uid)
                else:
                    nodes[parent_uid].children.insert(int(order), uid)

            case "update-block":
                # Does not affect structure for this interpreter.
                uid = a["block"]["uid"]
                if uid not in nodes:
                    raise AssertionError(f"update references missing uid {uid}")

            case "delete-block":
                uid = a["block"]["uid"]
                recursive_delete(uid)

            case other:
                raise AssertionError(f"unsupported action: {other}")

    return nodes


def test_generated_action_order_never_references_missing_parents():
    """
    Extreme: mixed create/move/update/delete where:
    - new parent is created
    - existing blocks are moved under it
    - an old ancestor is deleted (recursive)

    The ordered actions must be executable without referencing missing parents.
    """
    page_uid = "page"

    keep = _eb("K", "Keep", 0, "A")
    drop = _eb("D", "Drop", 1, "A")
    a = _eb("A", "OldParent", 0, page_uid, children=[keep, drop])
    # Add extra unmatched existing blocks to disable the "position fallback" matcher
    # (it only runs when both sides have <= 3 unmatched items). This forces the
    # diff to exercise create+move+delete ordering instead of "rename/update".
    z1 = _eb("Z1", "Extra1", 1, page_uid)
    z2 = _eb("Z2", "Extra2", 2, page_uid)

    # New structure:
    # - Create a new parent "NewParent"
    # - Move existing "Keep" under "NewParent"
    # - Delete OldParent subtree and other unmatched blocks
    new_parent = _nb("NP", "NewParent", page_uid)
    new_keep = _nb("newK", "Keep", "NP")

    diff = diff_block_trees([a, z1, z2], [new_parent, new_keep], page_uid)
    actions = generate_batch_actions(diff)

    # Executes without dangling references (create first, deletes last).
    nodes = _apply_actions_with_recursive_delete(page_uid, [a, z1, z2], actions)

    assert "NP" in nodes
    assert "K" in nodes  # preserved existing uid
    assert nodes["K"].parent_uid == "NP"
    assert "A" not in nodes
