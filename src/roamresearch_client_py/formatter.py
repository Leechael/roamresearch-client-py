from typing import cast
import re

REF_PATTERN = r'\(\(([a-zA-Z0-9_-]*?)\)\)'

def extract_ref(text):
    return re.findall(REF_PATTERN, text)

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
