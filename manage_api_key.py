"""
manage_api_keys.py — create, list, and revoke SnapKnow API keys.

This is how YOU (the project owner) hand out access to your API — there is
deliberately no way for a caller to get a key through the API itself, so
every key that exists was issued on purpose by you.

Usage:
    python manage_api_keys.py create "Person's Name"
    python manage_api_keys.py list
    python manage_api_keys.py revoke sk_xxxxxxxxxxxxxxxx

Keys are stored in api_keys.json, next to this file — the same file
api_server.py reads on every request. Add api_keys.json to .gitignore;
it should never be committed or shared publicly.
"""

import json
import os
import secrets
import sys
from datetime import datetime

API_KEYS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_keys.json")


def load_keys() -> dict:
    if not os.path.exists(API_KEYS_FILE):
        return {}
    with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_keys(keys: dict):
    with open(API_KEYS_FILE, "w", encoding="utf-8") as f:
        json.dump(keys, f, indent=2)


def cmd_create(owner: str):
    new_key = "sk_" + secrets.token_urlsafe(24)
    keys = load_keys()
    keys[new_key] = {
        "owner": owner,
        "created": datetime.now().isoformat(timespec="seconds"),
        "last_used": None,
        "request_count": 0,
        "active": True,
    }
    save_keys(keys)
    print(f"\nNew SnapKnow API key created for '{owner}':\n")
    print(f"  {new_key}\n")
    print("Give this key to them directly (e.g. over a private chat) — it is")
    print("shown here once and won't be printed again. They should send it in")
    print("the 'X-API-Key' header on every request to your API.\n")


def cmd_list():
    keys = load_keys()
    if not keys:
        print("No API keys created yet. Run: python manage_api_keys.py create \"Name\"")
        return
    print(f"{'KEY':<18}{'OWNER':<25}{'CREATED':<21}{'LAST USED':<21}{'REQUESTS':<10}{'STATUS'}")
    print("-" * 105)
    for key, info in keys.items():
        masked = "sk_..." + key[-6:]
        status = "active" if info.get("active", True) else "REVOKED"
        last_used = info.get("last_used") or "never"
        print(
            f"{masked:<18}{info.get('owner', ''):<25}{info.get('created', ''):<21}"
            f"{last_used:<21}{info.get('request_count', 0):<10}{status}"
        )


def cmd_revoke(key: str):
    keys = load_keys()
    if key not in keys:
        # Allow revoking by just the last 6 characters shown in `list`, since
        # that's the only part of the key anyone but the owner ever sees again.
        matches = [k for k in keys if k.endswith(key)]
        if len(matches) == 1:
            key = matches[0]
        elif len(matches) > 1:
            print("That suffix matches more than one key — paste the full key instead.")
            return
        else:
            print(f"No such key: {key}")
            return
    keys[key]["active"] = False
    save_keys(keys)
    print(f"Key for '{keys[key].get('owner', '')}' has been revoked.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "create" and len(sys.argv) >= 3:
        cmd_create(" ".join(sys.argv[2:]))
    elif cmd == "list":
        cmd_list()
    elif cmd == "revoke" and len(sys.argv) >= 3:
        cmd_revoke(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()