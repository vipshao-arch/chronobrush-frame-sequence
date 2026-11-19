"""Package only the MP4-free runtime data assets loaded by ChronoBrush."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "public" / "assets"
OUTPUT = ROOT / "chronobrush-runtime-assets.zip"
MANIFEST = ROOT / "chronobrush-runtime-assets.sha256.json"

FILES = (
    "base.webp",
    "reference.webp",
    "motion-mask.webp",
    "timeline.json",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    timeline = json.loads((SOURCE / "timeline.json").read_text(encoding="utf-8"))
    declared = {
        timeline["base"]["file"],
        timeline["sheet"]["file"],
        timeline["mask"]["file"],
        "timeline.json",
    }
    assert declared == set(FILES), (declared, FILES)
    for name in FILES:
        path = SOURCE / name
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = {
        "archive": OUTPUT.name,
        "contents": f"assets/{{{','.join(FILES)}}}",
        "sourceFrames": timeline["frameCount"],
        "fps": timeline["fps"],
        "commands": len(timeline["commands"]),
        "files": [],
    }
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name in FILES:
            path = SOURCE / name
            archive.write(path, f"assets/{name}")
            manifest["files"].append({
                "path": f"assets/{name}",
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            })
    manifest["archiveBytes"] = OUTPUT.stat().st_size
    manifest["archiveSha256"] = digest(OUTPUT)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "archive": str(OUTPUT),
        "archiveBytes": manifest["archiveBytes"],
        "files": FILES,
    }, indent=2))


if __name__ == "__main__":
    main()
