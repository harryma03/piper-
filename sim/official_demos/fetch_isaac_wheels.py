#!/usr/bin/env python3
"""Resume Isaac Sim wheels from NVIDIA and verify their published SHA256."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import parse_qs, unquote, urljoin, urlsplit, urlunsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "records/isaac_stack/wheels"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.append(dict(attrs).get("href", ""))


def fetch(package):
    index = f"https://pypi.nvidia.com/{package}/"
    for attempt in range(5):
        try:
            with urlopen(index, timeout=30) as response:
                html = response.read().decode()
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(attempt + 1)
    parser = Links()
    parser.feed(html)
    candidates = [urljoin(index, href) for href in parser.links
                  if "5.1.0.0" in href and "cp311" in href and "manylinux" in href and "x86_64.whl" in href]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one wheel for {package}, found {candidates}")
    parsed = urlsplit(candidates[0])
    expected = parse_qs(parsed.fragment)["sha256"][0]
    url = urlunsplit(parsed._replace(fragment=""))
    filename = unquote(Path(parsed.path).name)
    destination = OUTPUT / filename

    def digest():
        sha = hashlib.sha256()
        with destination.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                sha.update(block)
        return sha.hexdigest()

    if destination.exists() and digest() == expected:
        return {"file": filename, "sha256": expected, "url": url}
    print(f"Downloading {filename}", flush=True)
    for attempt in range(6):
        completed = subprocess.run([
            "curl", "--http1.1", "--fail", "--location", "--connect-timeout", "20",
            "--speed-limit", "1000", "--speed-time", "45", "--continue-at", "-",
            "--silent", "--show-error", "--output", str(destination), url,
        ])
        if completed.returncode == 0 and digest() == expected:
            print(f"Verified {filename}", flush=True)
            return {"file": filename, "sha256": expected, "url": url}
        if completed.returncode == 0:
            raise RuntimeError(f"SHA256 mismatch: {destination}")
        print(f"Retrying {filename}, attempt {attempt + 1}", flush=True)
        time.sleep(attempt + 1)
    raise RuntimeError(f"Download failed: {filename}")


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    packages = [f"isaacsim-{name}" for name in ("gui", "sensor", "robot", "replicator", "kernel", "asset")]
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(fetch, packages))
    (OUTPUT / "NVIDIA_WHEELS.json").write_text(json.dumps(files, indent=2) + "\n")
