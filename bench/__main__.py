"""`python -m bench` 的兜底入口：等价于 `python -m bench.edge`。

因为根目录同时存在 `bench.py` 和 `bench/` 包，`-m bench` 会命中包，
这里转发给 edge.py，避免"python -m bench --num-seqs ..."静默失效。
"""

import sys

from .edge import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
