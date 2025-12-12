#!/usr/bin/env bash
set -euo pipefail
source venv/bin/activate
pytest
python3 -m compileall -q . -x '(^|/)(ARCHIVE_BROKEN)(/|$)'
deactivate
echo "OK: pytest + compileall (excluding ARCHIVE_BROKEN)"
