"""Standalone updater entrypoint; must not be the running GUI binary."""
from brawlia_v2.installer import main

if __name__ == "__main__":
    raise SystemExit(main())
