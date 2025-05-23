from typing import cast, Union
from itertools import chain
import uuid
import logging

import mistune

from .RoamClient import create_block, Block

parse = mistune.create_markdown(renderer=None)
logger = logging.getLogger(__name__)


def parse_file(path_str: str):
    with open(path_str) as fp:
        return parse(fp.read())


def gen_uid():
    return uuid.uuid4().hex


def ast_to_inline(ast: dict):
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
            return ast_to_inline(ast['children'][0])
    logger.warn(f'unsupported inline type: {ast["type"]}')
    return ""


def ast_to_block(
        ast: dict,
        uid: Union[str, None] = None,
        pid: Union[str, None] = None
    ):
    match ast['type']:
        case 'heading':
            assert len(ast['children']) == 1
            level = ast['attrs']['level']
            items = [ast_to_inline(i) for i in ast['children']]
            blk = create_block(''.join(items), pid, gen_uid())
            if level <= 3:
                blk['block']['heading'] = level
            return [blk]

        case 'list':
            if not pid:
                blk = create_block("", None, gen_uid())
                nested = [ast_to_block(i, pid=blk['block']['uid']) for i in ast['children']]
                return [blk] + list(chain(*nested))
            else:
                nested = [ast_to_block(i, pid=pid) for i in ast['children']]
                return list(chain(*nested))

        case 'list_item':
            ret = ast_to_block(ast['children'][0], uid=uid, pid=pid)
            cur = ret[0]
            nested = [ast_to_block(i, pid=cur['block']['uid']) for i in ast['children'][1:]]
            # return ast_to_block(ast['children'][0], block_obj)
            return [cur] + list(chain(*nested))

        case 'block_text':
            items = [ast_to_inline(i) for i in ast['children']]
            return [create_block("".join(items), pid, gen_uid())]

        case 'paragraph':
            items = [ast_to_inline(i) for i in ast['children']]
            return [create_block("".join(items), pid, gen_uid())]

        case 'blank_line':
            # return [create_block("", pid, gen_uid())]
            return []

    logger.warn(f"unsupported block type: {ast['type']}")

    return []


def gfm_to_batch_actions(raw: str, pid):
    actions = []
    for blk in parse(raw):
        lst = ast_to_block(cast(dict, blk), pid=pid)
        if lst:
            actions.extend(lst)
    return actions
