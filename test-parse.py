from glob import glob

from src.roamresearch_client_py.gfm_to_roam import gfm_to_blocks, parse


if __name__ == '__main__':
    for fn in glob("storage/*.md"):
        print(f"TEST: parse {fn}")
        with open(fn) as fp:
            md = fp.read()
            blks = gfm_to_blocks(md, '<virtual_pid>')
