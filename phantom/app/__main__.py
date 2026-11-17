import os
import sys

# --res must be known before anything imports phantom.config (resolution is fixed at import time)
if "--res" in sys.argv:
    i = sys.argv.index("--res")
    if i + 1 < len(sys.argv):
        os.environ["PHANTOM_RES"] = sys.argv[i + 1]

from .server import main  # noqa: E402

main()
