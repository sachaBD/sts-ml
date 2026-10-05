"""Minimal Google Drive public-folder helpers (stdlib only): list a folder, download a file into memory."""
from __future__ import annotations

import re
import time
import urllib.error
import urllib.request

LIST_URL = "https://drive.google.com/embeddedfolderview?id={}"
FILE_URL = "https://drive.google.com/uc?export=download&id={}"
UA = {"User-Agent": "Mozilla/5.0 (megacrit_dump research tool)"}
ENTRY = re.compile(r'<div class="flip-entry" id="entry-([^"]+)".*?href="([^"]*)".*?class="flip-entry-title">([^<]*)<', re.S)
LIST_CAP = 5500  # observed: embeddedfolderview never returns more than this many entries


class Throttled(Exception):
    """Drive refused service (429/403/quota page); callers should stop, not retry harder."""


def parse_listing(html: str) -> list[tuple[str, str, bool]]:
    """-> [(id, name, is_folder)]"""
    out = []
    for fid, href, name in ENTRY.findall(html):
        out.append((fid, name.replace("&amp;", "&"), "/drive/folders/" in href))
    return out


def fetch(url: str, timeout: float = 60, retries: int = 4, sleep=time.sleep) -> bytes:
    """GET with exponential backoff on transient errors. Raises Throttled on 429/403 (after backoff) or a quota page."""
    delay = 2.0
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in (403, 429, 500, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        if attempt < retries:
            sleep(delay)
            delay *= 2
    if isinstance(last, urllib.error.HTTPError) and last.code in (403, 429):
        raise Throttled(f"HTTP {last.code} for {url}")
    raise last  # type: ignore[misc]


def download(file_id: str, **kw) -> bytes:
    data = fetch(FILE_URL.format(file_id), **kw)
    if data[:2] != b"\x1f\x8b":  # an HTML page (quota exceeded / virus-scan interstitial) instead of gzip
        text = data[:2000].decode("utf-8", "replace").lower()
        if "quota" in text or "too many" in text or "unusual traffic" in text:
            raise Throttled(f"quota page for {file_id}")
        raise ValueError(f"not gzip: {file_id} ({len(data)} bytes)")
    return data
