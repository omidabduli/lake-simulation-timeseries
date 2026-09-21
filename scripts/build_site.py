#!/usr/bin/env python3
"""Assemble the static GitHub Pages site (in-browser build, powered by stlite).

    python3 scripts/build_site.py              # build into _site/
    python3 scripts/build_site.py --serve      # build, then preview at http://127.0.0.1:8000/

Output layout:
    index.html, 404.html, favicon.png   host page (loads stlite + Pyodide from jsDelivr)
    app/manifest.json                   entrypoint, browser requirements, file list
    app/...                             Python app files that stlite mounts into the
                                        browser's virtual file system

Uses only the Python standard library, so it runs in CI without installs.
"""
from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import http.server
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
STATIC_FILES = ("index.html", "404.html", "favicon.png")
DEMO_WORKBOOK = "Example/Lake_Time_Series_Forecasting_Demo_2000_Rows.xlsx"


def app_files() -> dict[str, Path]:
    """Map each path in the browser's virtual file system to its source file.

    New modules in modules/ are picked up automatically; any other data file
    the app opens at runtime must be added here.
    """
    files = {
        "entrypoint.py": WEB / "entrypoint.py",
        "app.py": ROOT / "app.py",
        ".streamlit/config.toml": ROOT / ".streamlit" / "config.toml",
        DEMO_WORKBOOK: ROOT / DEMO_WORKBOOK,
    }
    for module in sorted((ROOT / "modules").glob("*.py")):
        files[f"modules/{module.name}"] = module
    return files


def served_path(mount_path: str) -> str:
    """URL path for a mounted file. Leading dots are dropped from each segment
    because upload-pages-artifact excludes hidden files from the deployment."""
    return "app/" + "/".join(part.lstrip(".") for part in mount_path.split("/"))


def read_requirements(path: Path) -> list[str]:
    lines = (line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line]


def check_python(source: Path) -> None:
    """Fail the build on syntax errors. The entrypoint uses top-level await,
    which stlite supports."""
    flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT if source.name == "entrypoint.py" else 0
    compile(source.read_text(encoding="utf-8"), str(source), "exec", flags=flags, dont_inherit=True)


def git_commit() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build(out: Path) -> None:
    if out.exists():
        if any(out.iterdir()) and not (out / "app" / "manifest.json").is_file():
            sys.exit(f"Refusing to overwrite {out}: it is not empty and not a previous build.")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for name in STATIC_FILES:
        shutil.copy2(WEB / name, out / name)

    served: dict[str, str] = {}
    urls: dict[str, str] = {}
    for mount_path, source in app_files().items():
        if not source.is_file():
            sys.exit(f"Missing app file: {source.relative_to(ROOT)}")
        if source.suffix == ".py":
            check_python(source)
        url = served_path(mount_path)
        if url in served:
            sys.exit(f"{mount_path} and {served[url]} would be served from the same URL {url}")
        served[url] = mount_path
        target = out / url
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        urls[mount_path] = f"{url}?v={digest}"  # content hash busts browser caches after a deploy

    manifest = {
        "entrypoint": "entrypoint.py",
        "requirements": read_requirements(WEB / "requirements.txt"),
        "files": urls,
        "commit": git_commit(),
    }
    (out / "app" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"Built {out} ({len(urls)} app files, {size / 1024:.0f} KiB)")


def serve(out: Path, port: int) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"Serving {out} at http://127.0.0.1:{port}/  (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "_site", help="output directory (default: _site)")
    parser.add_argument("--serve", action="store_true", help="serve the built site locally after building")
    parser.add_argument("--port", type=int, default=8000, help="port for --serve (default: 8000)")
    args = parser.parse_args()

    build(args.out)
    if args.serve:
        serve(args.out, args.port)


if __name__ == "__main__":
    main()
