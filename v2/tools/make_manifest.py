"""Release manifest generator run on the trusted GitHub Windows runner."""
from pathlib import Path
import json
import sys
import hashlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brawlia_v2 import __version__


def main():
    app_path, output_path = map(Path, sys.argv[1:3])
    digest = hashlib.sha256(app_path.read_bytes()).hexdigest()
    manifest = {"version": __version__, "filename": app_path.name,
                "sha256": digest, "size": app_path.stat().st_size}
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
