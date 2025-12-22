from typing import cast, List, Dict, Any, Literal
from itertools import chain
import uuid
import logging

import mistune

from .client import create_block
from .structs import Block, BlockRef

parse = mistune.create_markdown(renderer=None, plugins=['table'])
logger = logging.getLogger(__name__)


def parse_file(path_str: str):
    with open(path_str) as fp:
        return parse(fp.read())


def gen_uid():
    return uuid.uuid4().hex


def ast_to_inline(ast: dict) -> str:
    match ast['type']:
        case 'text':
            if ast.get('attrs', {}).get('url'):
                return f"[{ast['raw']}]({ast['attrs']['url']})"
            return ast['raw']
        case 'codespan':
            if "children" in ast:
                text = "".join([ast_to_inline(i) for i in ast["children"]])
                return f'`{text}`'
            else:
                return f'`{ast["raw"]}`'
        case "strong":
            if "children" in ast:
                text = "".join([ast_to_inline(i) for i in ast["children"]])
                return f"**{text}**"
            else:
                return f"**{ast['raw']}**"
        case "emphasis":
            if "children" in ast:
                text = "".join([ast_to_inline(i) for i in ast["children"]])
                return f"*{text}*"
            else:
                return f"*{ast['raw']}*"
        case "link":
            text = ast_to_inline(ast['children'][0])
            url = ast.get("attrs", {}).get("url")
            # TODO escape text and url to ensure not breaks
            if url:
                return f"[{text}]({url})"
            else:
                return text
        case 'softbreak':
            return "\n"
        case 'linebreak':
            return "\n\n"
    logger.warn(f'unsupported inline type: {ast["type"]}')
    return ""

def ast_to_block(
        ast: dict,
        parent_ref: BlockRef,
        prefix: str | None = None
    ) -> list[Block]:
    match ast['type']:
        # NOTE: RoamResearch only supports heading up to level 3
        case 'heading':
            items = [ast_to_inline(i) for i in ast['children']]
            blk = Block(''.join(items), parent_ref)
            blk.heading = ast['attrs']['level']
            return [blk]

        case 'list':
            nested = [ast_to_block(i, parent_ref) for i in ast['children']]
            lst = []
            is_ordered = ast.get('attrs', {}).get('ordered', False)
            for idx, i in enumerate(ast['children']):
                prefix = ''
                if is_ordered:
                    prefix = f'{idx+1}. '
                blks = ast_to_block(i, parent_ref, prefix)
                lst.extend(blks)
            return lst

        case 'list_item':
            cur, = ast_to_block(ast['children'][0], parent_ref)
            cur.text = f'{prefix}{cur.text}'
            nested = [ast_to_block(i, cur.ref) for i in ast['children'][1:]]
            return [cur] + list(chain(*nested))

        case 'block_text':
            items = [ast_to_inline(i) for i in ast['children']]
            return [Block("".join(items), parent_ref)]

        case 'paragraph':
            items = [ast_to_inline(i) for i in ast['children']]
            return [Block("".join(items), parent_ref)]

        case 'blank_line':
            # return [create_block("", pid, gen_uid())]
            return []
        
        case 'table':
            # Typical table structure from mistune AST will contains two children: table_head and table_body
            table_block = Block(text="{{[[table]]}}", parent_ref=parent_ref, open=False)
            lst = [table_block]
            for i in ast["children"]:
                children = ast_to_block(i, table_block.ref)
                lst.extend(children)
            return lst

        case 'table_head':
            lst = []
            ref = parent_ref
            for i in ast['children']:
                child, = ast_to_block(i, ref)
                lst.append(child)
                ref = child.ref
            return lst

        case 'table_body':
            lst = []
            for i in ast['children']:
                cells = ast_to_block(i, parent_ref)
                lst.extend(cells)
            return lst
        
        case 'table_row':
            lst = []
            ref = parent_ref
            for i in ast['children']:
                cell, = ast_to_block(i, ref)
                lst.append(cell)
                ref = cell.ref
            return lst

        case 'table_cell':
            items = [ast_to_inline(i) for i in ast['children']]
            return [Block("".join(items), parent_ref)]
    
        case 'block_code':
            lang = ast.get('attrs', {}).get('info', '')
            code = ast.get('raw', '')
            return [Block(f"```{lang}\n{code}\n```", parent_ref)]

        case 'thematic_break':
            return [Block("---", parent_ref)]

        case 'block_quote':
            if len(ast.get("children", [])) > 1 or ast["children"][0].get("type") != 'paragraph':
                logger.warn(f"Unexpected AST for block type block_quote: {ast}")
                return []
            blk, = ast_to_block(ast["children"][0], parent_ref)
            blk.text = f"> {blk.text}"
            return [blk]

    logger.warn(f"unsupported block type: {ast['type']}")

    return []


def gfm_to_blocks(raw: str, pid: str):
    blocks = []
    ref = BlockRef(block_uid=pid)

    parsed = parse(raw)
    pid_stack: List[Dict[str, Any]] = [{'level': 0, 'ref': ref}]

    for blk in parsed:
        blk = cast(dict, blk)
        if (blk['type']) == 'heading':
            level = blk['attrs']['level']
            if level == 1:
                continue
            while pid_stack[-1]['level'] >= level:
                pid_stack.pop()
        if blk['type'] == 'thematic_break':
            continue
        if blk['type'] == 'list':
            try:
                lst = ast_to_block(blk, blocks[-1].ref)
            except IndexError:
                lst = ast_to_block(blk, ref)
        else:
            lst = ast_to_block(blk, pid_stack[-1]['ref'])
        if not lst:
            continue
        if (blk['type']) == 'heading':
            pid_stack.append({'level': blk['attrs']['level'], 'ref': lst[0].ref})
        blocks.extend(lst)
    return blocks


def gfm_to_batch_actions(raw: str, pid: str):
    blocks = gfm_to_blocks(raw, pid)
    return [b.to_create_action() for b in blocks]