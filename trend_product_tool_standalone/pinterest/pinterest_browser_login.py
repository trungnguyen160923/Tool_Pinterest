from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pinterest.image_crawler.pinterest_browser_login import main


if __name__ == "__main__":
    raise SystemExit(main())
