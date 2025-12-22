"""
Diff algorithm for comparing Roam blocks and generating minimal update actions.

This module provides functionality to:
1. Parse existing blocks from Roam API responses
2. Match new blocks with existing blocks by content similarity
3. Generate minimal diff actions (create, update, move, delete)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from .structs import Block


@dataclass
class ExistingBlock:
    """Represents a block fetched from Roam API."""
    uid: str
    text: str
    order: int
    heading: int | None
    children: list[ExistingBlock] = field(default_factory=list)
    parent_uid: str | None = None

    @classmethod
    def from_roam_dict(cls, d: dict, parent_uid: str | None = None) -> ExistingBlock:
        """Parse a Roam API response dict into an ExistingBlock."""
        children_raw = d.get(':block/children', [])
        children_sorted = sorted(children_raw, key=lambda x: x.get(':block/order', 0))

        uid = d.get(':block/uid', '')
        children = [cls.from_roam_dict(c, parent_uid=uid) for c in children_sorted]

        return cls(
            uid=uid,
            text=d.get(':block/string', ''),
            order=d.get(':block/order', 0),
            heading=d.get(':block/heading'),
            children=children,
            parent_uid=parent_uid
        )

    def flatten(self) -> list[ExistingBlock]:
        """Flatten this block and all descendants into a list."""
        result = [self]
        for child in self.children:
            result.extend(child.flatten())
        return result


@dataclass
class DiffResult:
    """Result of diffing two block trees."""
    creates: list[dict] = field(default_factory=list)
    updates: list[dict] = field(default_factory=list)
    moves: list[dict] = field(default_factory=list)
    deletes: list[dict] = field(default_factory=list)
    preserved_uids: set[str] = field(default_factory=set)

    def stats(self) -> dict:
        """Return statistics about the diff."""
        return {
            'creates': len(self.creates),
            'updates': len(self.updates),
            'moves': len(self.moves),
            'deletes': len(self.deletes),
            'preserved': len(self.preserved_uids)
        }

    def is_empty(self) -> bool:
        """Return True if no changes are needed."""
        return not (self.creates or self.updates or self.moves or self.deletes)


def parse_existing_blocks(roam_response: dict | list) -> list[ExistingBlock]:
    """
    Parse Roam API response into a list of ExistingBlock trees.

    Args:
        roam_response: Either a page dict with :block/children,
                       or a list of block dicts

    Returns:
        List of top-level ExistingBlock objects
    """
    if isinstance(roam_response, dict):
        # Page response with children
        children = roam_response.get(':block/children', [])
        parent_uid = roam_response.get(':block/uid')
        children_sorted = sorted(children, key=lambda x: x.get(':block/order', 0))
        return [ExistingBlock.from_roam_dict(c, parent_uid=parent_uid) for c in children_sorted]
    elif isinstance(roam_response, list):
        # List of blocks
        return [ExistingBlock.from_roam_dict(b) for b in roam_response]
    return []


def flatten_existing_blocks(blocks: list[ExistingBlock]) -> list[ExistingBlock]:
    """Flatten nested ExistingBlock tree into a flat list."""
    result = []
    for block in blocks:
        result.extend(block.flatten())
    return result


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.strip()


def normalize_for_matching(text: str) -> str:
    """
    Normalize text for matching, removing formatting differences.

    This handles cases where:
    - Markdown lists add "1. ", "2. " prefixes
    - Extra whitespace differences
    """
    import re
    text = text.strip()
    # Remove list number prefixes like "1. ", "2. ", etc.
    text = re.sub(r'^\d+\.\s+', '', text)
    return text


def match_blocks(
    existing: list[ExistingBlock],
    new_blocks: list['Block']
) -> dict[str, str]:
    """
    Match new blocks to existing blocks by content similarity.

    Strategy:
    1. Exact text match (greedy, prefer same position)
    2. Normalized text match (ignoring list prefixes)
    3. Position-based fallback for remaining blocks (only if text is similar)

    Args:
        existing: List of existing blocks from Roam
        new_blocks: List of new blocks from markdown

    Returns:
        Dict mapping new_block_uid -> existing_uid for matched pairs
    """
    matches: dict[str, str] = {}
    used_existing: set[str] = set()

    # Build index: normalized_text -> list of existing blocks
    existing_by_text: dict[str, list[ExistingBlock]] = defaultdict(list)
    existing_by_normalized: dict[str, list[ExistingBlock]] = defaultdict(list)
    for eb in existing:
        norm_text = normalize_text(eb.text)
        existing_by_text[norm_text].append(eb)
        # Also index by normalized (list prefix removed)
        match_text = normalize_for_matching(eb.text)
        existing_by_normalized[match_text].append(eb)

    # Phase 1: Exact text matches with position preference
    for idx, new_block in enumerate(new_blocks):
        norm_text = normalize_text(new_block.text)
        candidates = [e for e in existing_by_text.get(norm_text, [])
                      if e.uid not in used_existing]

        if candidates:
            # Prefer candidate closest to same position
            best = min(candidates, key=lambda e: abs(e.order - idx))
            matches[new_block.ref.block_uid] = best.uid
            used_existing.add(best.uid)

    # Phase 2: Normalized text matches (handles list prefix differences)
    for idx, new_block in enumerate(new_blocks):
        if new_block.ref.block_uid in matches:
            continue

        match_text = normalize_for_matching(new_block.text)
        candidates = [e for e in existing_by_normalized.get(match_text, [])
                      if e.uid not in used_existing]

        if candidates:
            best = min(candidates, key=lambda e: abs(e.order - idx))
            matches[new_block.ref.block_uid] = best.uid
            used_existing.add(best.uid)

    # Phase 3: Position-based fallback for truly unmatched blocks
    # Only match if there's exactly one unmatched on each side at same level
    # This is conservative to avoid wrong matches
    unmatched_new = [b for b in new_blocks if b.ref.block_uid not in matches]
    unmatched_existing = [e for e in existing if e.uid not in used_existing]

    # Only use position fallback if very few unmatched (likely edits)
    if len(unmatched_new) <= 3 and len(unmatched_existing) <= 3:
        unmatched_existing_sorted = sorted(unmatched_existing, key=lambda e: e.order)
        for idx, new_block in enumerate(unmatched_new):
            if idx < len(unmatched_existing_sorted):
                exist_block = unmatched_existing_sorted[idx]
                matches[new_block.ref.block_uid] = exist_block.uid
                used_existing.add(exist_block.uid)

    return matches


def diff_block_trees(
    existing: list[ExistingBlock],
    new_blocks: list['Block'],
    parent_uid: str
) -> DiffResult:
    """
    Compute minimal diff between existing and new block trees.

    This function generates the minimum set of actions needed to transform
    the existing blocks into the new blocks while preserving UIDs where possible.

    Args:
        existing: Current blocks from Roam (may be nested tree)
        new_blocks: Desired blocks from markdown (flat list with parent_ref)
        parent_uid: UID of the parent (page or block)

    Returns:
        DiffResult with create, update, move, delete actions
    """
    result = DiffResult()

    # Flatten existing blocks for comparison
    existing_flat = flatten_existing_blocks(existing)

    # Step 1: Match blocks by content
    matches = match_blocks(existing_flat, new_blocks)

    # Build reverse mapping: existing_uid -> new_block
    existing_to_new: dict[str, 'Block'] = {}
    for new_uid, exist_uid in matches.items():
        for nb in new_blocks:
            if nb.ref.block_uid == new_uid:
                existing_to_new[exist_uid] = nb
                break

    # Build existing uid -> ExistingBlock mapping
    existing_by_uid = {eb.uid: eb for eb in existing_flat}

    # Group matched new blocks by their EXISTING parent (preserve structure)
    # Key: existing parent_uid, Value: list of (new_block, exist_uid) in markdown order
    matched_by_existing_parent: dict[str, list[tuple['Block', str]]] = {}
    for new_block in new_blocks:
        new_uid = new_block.ref.block_uid
        if new_uid in matches:
            exist_uid = matches[new_uid]
            exist_block = existing_by_uid[exist_uid]
            exist_parent = exist_block.parent_uid or parent_uid
            if exist_parent not in matched_by_existing_parent:
                matched_by_existing_parent[exist_parent] = []
            matched_by_existing_parent[exist_parent].append((new_block, exist_uid))

    # Step 2: Process matched blocks (check for updates/moves)
    for new_block in new_blocks:
        new_uid = new_block.ref.block_uid

        if new_uid in matches:
            exist_uid = matches[new_uid]
            exist_block = existing_by_uid[exist_uid]
            exist_parent = exist_block.parent_uid or parent_uid
            result.preserved_uids.add(exist_uid)

            # Check for text/heading changes -> update-block
            needs_update = False
            update_action = {
                "action": "update-block",
                "block": {"uid": exist_uid}
            }

            if normalize_text(new_block.text) != normalize_text(exist_block.text):
                update_action["block"]["string"] = new_block.text
                needs_update = True

            new_heading = getattr(new_block, '_heading', None)
            if new_heading != exist_block.heading:
                if new_heading is not None:
                    update_action["block"]["heading"] = new_heading
                    needs_update = True
                elif exist_block.heading is not None:
                    # Remove heading (set to 0 or omit - Roam API specific)
                    update_action["block"]["heading"] = 0
                    needs_update = True

            if needs_update:
                result.updates.append(update_action)

            # Check for order changes within SAME parent -> move-block
            # Calculate expected order: position in markdown among siblings with same existing parent
            siblings = matched_by_existing_parent.get(exist_parent, [])
            expected_order = next(
                (i for i, (nb, _) in enumerate(siblings) if nb.ref.block_uid == new_uid),
                exist_block.order
            )

            # Only move if order changed (parent stays the same)
            if exist_block.order != expected_order:
                result.moves.append({
                    "action": "move-block",
                    "block": {"uid": exist_uid},
                    "location": {
                        "parent-uid": exist_parent,
                        "order": expected_order
                    }
                })
        else:
            # No match -> create-block
            # For new blocks, use the page as parent (simple approach)
            # TODO: Could try to find appropriate parent based on markdown structure
            create_action = {
                "action": "create-block",
                "location": {
                    "parent-uid": parent_uid,
                    "order": "last"
                },
                "block": {
                    "uid": new_block.ref.block_uid,
                    "string": new_block.text
                }
            }

            new_heading = getattr(new_block, '_heading', None)
            if new_heading is not None:
                create_action["block"]["heading"] = new_heading

            new_open = getattr(new_block, 'open', True)
            if new_open is not None:
                create_action["block"]["open"] = new_open

            result.creates.append(create_action)

    # Step 3: Find unmatched existing blocks -> delete-block
    matched_existing_uids = set(matches.values())
    for exist_block in existing_flat:
        if exist_block.uid not in matched_existing_uids:
            result.deletes.append({
                "action": "delete-block",
                "block": {"uid": exist_block.uid}
            })

    return result


def generate_batch_actions(diff: DiffResult) -> list[dict]:
    """
    Convert DiffResult to ordered batch actions for Roam API.

    Order of operations:
    1. Deletes (bottom-up to avoid orphan issues)
    2. Moves (reposition existing blocks)
    3. Creates (top-down to ensure parents exist)
    4. Updates (order doesn't matter)

    Args:
        diff: The DiffResult to convert

    Returns:
        Ordered list of action dicts ready for batch_actions API
    """
    actions = []

    # 1. Deletes first (reverse order to delete children before parents)
    actions.extend(reversed(diff.deletes))

    # 2. Moves
    actions.extend(diff.moves)

    # 3. Creates (in order, so parents before children)
    actions.extend(diff.creates)

    # 4. Updates last
    actions.extend(diff.updates)

    return actions
