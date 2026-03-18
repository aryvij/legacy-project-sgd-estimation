import sys, pathlib
# Ensure scripts/src/ is on sys.path so that "from core.…" imports resolve
_src = str(pathlib.Path(__file__).resolve().parent)
if _src not in sys.path:
    sys.path.insert(0, _src)