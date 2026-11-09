"""Interactive terminal prompt for the optional, free API keys. Press Enter to skip any."""
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cinemind import config  # noqa: E402


def main():
    if not sys.stdin.isatty():
        print("Non-interactive shell: skipping key prompts (add keys later in the dashboard Settings tab).")
        return
    status = config.key_status()
    print("\nCineMind - optional FREE API keys (press Enter to skip / keep current)\n")
    updates = {}
    for k, desc in config.API_KEYS.items():
        mark = "set" if status[k] else "not set"
        v = getpass.getpass(f"  {k} [{mark}]  ({desc}): ").strip()
        if v:
            updates[k] = v
    if updates:
        config.save_env(updates)
        print(f"Saved {len(updates)} key(s) to {config.ENV_FILE}")
    else:
        print("No changes. You can add keys any time from the dashboard's Settings tab.")


if __name__ == "__main__":
    main()
