import sys
from pathlib import Path

# Support running as `python src/roamresearch_client_py`
if __package__ is None or __package__ == "":
    src_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(src_dir))
    from roamresearch_client_py.cli import main
else:
    from .cli import main


if __name__ == "__main__":
    main()
