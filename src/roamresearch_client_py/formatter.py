from typing import cast
import re

REF_PATTERN = r'\(\(([a-zA-Z0-9_-]*?)\)\)'


def extract_ref(text):
    return re.findall(REF_PATTERN, text)


def _get_sorted_children(block):
    """Get direct children of a block, sorted by order."""
    children = block.get(':block/children', [])
    if not children:
        return []
    return sorted(children, key=lambda x: x.get(':block/order', 0))


def _collect_children_flat(block, min_level=2, current_level=1):
    """
    Recursively collect children blocks, flattened.
    Only include blocks at min_level or deeper.
    current_level is the level of `block`, its children are at current_level + 1.
    """
    result = []
    direct_children = _get_sorted_children(block)
    child_level = current_level + 1

    for child in direct_children:
        if child_level >= min_level:
            result.append(child)
        result.extend(_collect_children_flat(child, min_level, child_level))

    return result


def _flatten_all_blocks(block):
    """Recursively flatten all blocks for reference resolution."""
    blocks = [block]
    for child in block.get(':block/children', []):
        blocks.extend(_flatten_all_blocks(child))
    return blocks


def _resolve_refs(text, all_blocks):
    """Replace block references ((uid)) with actual text."""
    refs = extract_ref(text)
    for ref in refs:
        node = next((b for b in all_blocks if b.get(':block/uid') == ref), None)
        if node:
            text = text.replace(f"(({ref}))", node.get(':block/string', ''))
    return text


def _format_block_text(block, all_blocks):
    """Format a single block's text with heading prefix if applicable."""
    text = block.get(':block/string', '')
    text = _resolve_refs(text, all_blocks)

    heading = block.get(':block/heading')
    if heading:
        prefix = '#' * heading
        return f"{prefix} {text}"
    return text


def _is_table_block(block):
    """Check if block is a Roam table."""
    text = block.get(':block/string', '')
    return text.strip() == '{{[[table]]}}'


def _is_code_block(block):
    """Check if block is a code block."""
    text = block.get(':block/string', '')
    return text.strip().startswith('```')


def _collect_row_cells(row_block, all_blocks):
    """
    Collect cells from a table row.
    In Roam tables, cells are horizontally nested (each cell is child of previous).
    """
    cells = [row_block.get(':block/string', '')]
    current = row_block

    while True:
        children = _get_sorted_children(current)
        if not children:
            break
        # Take first child as next cell in row
        current = children[0]
        cells.append(current.get(':block/string', ''))

    return cells


def _format_table(block, all_blocks):
    """Format a Roam table as GFM table."""
    rows = _get_sorted_children(block)
    if not rows:
        return ''

    table_data = []
    for row in rows:
        cells = _collect_row_cells(row, all_blocks)
        # Resolve refs in each cell
        cells = [_resolve_refs(c, all_blocks) for c in cells]
        table_data.append(cells)

    if not table_data:
        return ''

    # Determine column count
    max_cols = max(len(row) for row in table_data)

    # Pad rows to same length
    for row in table_data:
        while len(row) < max_cols:
            row.append('')

    lines = []
    # Header row
    lines.append('| ' + ' | '.join(table_data[0]) + ' |')
    # Separator
    lines.append('| ' + ' | '.join(['---'] * max_cols) + ' |')
    # Data rows
    for row in table_data[1:]:
        lines.append('| ' + ' | '.join(row) + ' |')

    return '\n'.join(lines)


def format_block_as_markdown(blocks, all_blocks=None):
    """
    Format Roam blocks as GFM markdown.

    - Blocks with :block/heading become headings (#, ##, ###)
    - Top-level blocks without heading become paragraphs
    - Level 1 children: headings or paragraphs
    - Level 2+ children become a flat bullet list
    - {{[[table]]}} blocks become GFM tables
    """
    if all_blocks is None:
        all_blocks = []
        for b in blocks:
            all_blocks.extend(_flatten_all_blocks(b))

    lines = []

    for block in blocks:
        # Handle table blocks
        if _is_table_block(block):
            if lines:
                lines.append('')
            lines.append(_format_table(block, all_blocks))
            lines.append('')
            continue

        text = _format_block_text(block, all_blocks)
        heading = block.get(':block/heading')

        if heading:
            # Heading: add blank line before if not first
            if lines:
                lines.append('')
            lines.append(text)
        else:
            # Paragraph
            lines.append(text)

        lines.append('')

        # Level 1 children: headings or paragraphs
        level1_children = _get_sorted_children(block)
        for child in level1_children:
            if _is_table_block(child):
                lines.append(_format_table(child, all_blocks))
                lines.append('')
                continue

            child_text = _format_block_text(child, all_blocks)
            child_heading = child.get(':block/heading')

            if child_heading:
                if lines:
                    lines.append('')
                lines.append(child_text)
            else:
                lines.append(child_text)

            lines.append('')

            # Level 2+ children: list items (except code blocks)
            deep_children = _collect_children_flat(child, min_level=2, current_level=1)
            if deep_children:
                list_items = []
                for deep_child in deep_children:
                    deep_text = deep_child.get(':block/string', '')
                    deep_text = _resolve_refs(deep_text, all_blocks)

                    if _is_code_block(deep_child):
                        # Flush pending list items
                        if list_items:
                            lines.extend(list_items)
                            list_items = []
                            lines.append('')
                        # Output code block as-is
                        lines.append(deep_text)
                        lines.append('')
                    else:
                        list_items.append(f"- {deep_text}")

                # Flush remaining list items
                if list_items:
                    lines.extend(list_items)
                    lines.append('')

    return '\n'.join(lines).strip()

def format_block(i, nodes, indent=0):
    lines = []
    text = i[':block/string']
    refs = extract_ref(text)
    if refs:
        for ref in refs:
            node = next((k for k in nodes if k[':block/uid'] == ref), None)
            if node:
                text = text.replace(f"(({ref}))", node[':block/string'])
        for ref in i.get(':block/refs', []):
            if ":block/string" in ref:
                text = text.replace(f"(({ref[':block/uid']}))", ref[':block/string'])
    if indent > 0:
        lines.append(f"{'='*indent}> {text}")
    else:
        lines.append(text)
    if ':block/children' in i and len(i[':block/children']) > 0:
        children = filter(lambda x: x is not None,
                          [next((k for k in nodes if k[':db/id'] == j[':db/id']), None) for j in i.get(':block/children', [])])
        sorted_nodes = sorted(children, key=lambda k: cast(dict, k).get(':block/order', 0))
        for j in sorted_nodes:
            lines.append(format_block(j, nodes, indent+2))
    return "\n".join(lines)
