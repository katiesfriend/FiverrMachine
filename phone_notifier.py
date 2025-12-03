#!/usr/bin/env python3
"""
phone_notifier.py

Sends a push-style notification to your phone using ntfy.sh (free service).

USAGE:
  - Choose a topic name (like 'shadowbridge-mykl') and subscribe to it
    in the ntfy Android app.
  - Set the NTFY_TOPIC environment variable, for example:
        export NTFY_TOPIC="shadowbridge-mykl"
  - Then call:
        python3 phone_notifier.py "Title text" "Body text"

This uses HTTP POST; no account or payment required.
"""

import os
import sys
from typing import Optional

import requests


def send_ntfy_notification(title: str, message: str, topic: Optional[str] = None) -> None:
    """Send a notification to ntfy.sh/<topic>."""
    topic = topic or os.getenv("NTFY_TOPIC")
    if not topic:
        raise RuntimeError("NTFY_TOPIC environment variable is not set, and no topic was provided.")

    url = f"https://ntfy.sh/{topic}"
    headers = {
        "Title": title,
    }
    resp = requests.post(url, headers=headers, data=message.encode("utf-8"), timeout=10)
    resp.raise_for_status()
    print(f"[NTFY] Sent notification to topic '{topic}'.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 phone_notifier.py \"Title\" \"Message body here\"")
        raise SystemExit(1)

    title = sys.argv[1]
    message = sys.argv[2] if len(sys.argv) > 2 else ""
    send_ntfy_notification(title, message)


if __name__ == "__main__":
    main()
