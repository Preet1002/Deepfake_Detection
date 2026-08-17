"""Download generated faces from a public generator demo page.

Each request to thispersondoesnotexist.com returns a freshly generated face, so
repeated requests build a sample from a generator that is not in the training
set - which is what an unseen-generator measurement needs.

Two things this is careful about:

  * It is somebody's free demo site. Requests are spaced by --delay, and the
    default sample size is modest. Do not lower the delay.
  * Responses are deduplicated by content hash. If the server or a proxy caches,
    you would otherwise collect one image a hundred times and compute a
    "detection rate" over a single picture without noticing.

    python -m scripts.collect_faces --count 100 --out data/unseen_generator
"""
from __future__ import annotations

import argparse
import hashlib
import io
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# The site root serves an HTML page; the image itself is a separate path.
DEFAULT_URL = "https://thispersondoesnotexist.com/random-person.jpeg"

# Some hosts reject the default urllib agent outright.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; academic-dataset-collection)"}


def fetch(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def image_size(payload: bytes) -> tuple[int, int] | None:
    """(width, height) if the bytes decode as an image, else None.

    Worth checking every response, not just the first: a 200 carrying an HTML
    landing page, a rate-limit notice, or a CAPTCHA is still a 200, and saving
    it under a .jpg name produces a folder that only fails much later at scoring
    time. Ask the decoder rather than trusting the status code.
    """
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
            return image.size
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--count", type=int, default=100,
                        help="number of DISTINCT images to collect")
    parser.add_argument("--out", default="data/unseen_generator")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="seconds between requests; please do not lower this")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    # Re-running should top up an existing folder, not restart it.
    for existing in out_dir.glob("*.jpg"):
        seen.add(hashlib.md5(existing.read_bytes()).hexdigest())
    if seen:
        print(f"{len(seen)} images already in {out_dir}; topping up to {args.count}")

    duplicates = failures = 0
    while len(seen) < args.count:
        try:
            payload = fetch(args.url, args.timeout)
        except (urllib.error.URLError, TimeoutError) as exc:
            failures += 1
            print(f"  request failed ({exc}); {failures} so far")
            if failures >= 10:
                raise SystemExit(
                    "Ten failures - stopping. Check the URL is reachable in a "
                    "browser, or collect the images by hand.")
            time.sleep(args.delay * 2)
            continue

        size = image_size(payload)
        if size is None:
            head = payload[:80].decode("utf-8", "replace").replace("\n", " ")
            raise SystemExit(
                f"The response is not an image ({len(payload):,} bytes starting "
                f"{head!r}).\n"
                f"URL: {args.url}\n"
                f"The site is probably serving an HTML page rather than the raw "
                f"image. Open the URL in a browser, right-click the face, copy "
                f"the image address, and pass it with --url.")

        digest = hashlib.md5(payload).hexdigest()
        if digest in seen:
            duplicates += 1
            if duplicates in (10, 50):
                print(f"  {duplicates} duplicate responses so far - the source "
                      f"may be caching. Distinct images: {len(seen)}")
            if duplicates >= 100:
                print(f"\nStopping: 100 duplicates, so the source is serving "
                      f"cached content rather than fresh samples.\n"
                      f"Collected {len(seen)} distinct images.")
                break
        else:
            seen.add(digest)
            (out_dir / f"face_{len(seen):03d}.jpg").write_bytes(payload)
            print(f"  {len(seen)}/{args.count}", end="\r", flush=True)

        time.sleep(args.delay)

    print(f"\n{len(seen)} distinct images in {out_dir} "
          f"({duplicates} duplicate responses, {failures} failed requests)")


if __name__ == "__main__":
    main()
