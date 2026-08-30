"""检查已部署服务的健康状态。"""

from __future__ import annotations

import argparse
import json
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/healthz")
    args = parser.parse_args()
    with urllib.request.urlopen(args.url, timeout=5) as response:
        print(json.loads(response.read().decode("utf-8")))


if __name__ == "__main__":
    main()
