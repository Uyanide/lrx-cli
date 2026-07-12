from __future__ import annotations

import argparse
import asyncio
import tempfile
import time
from pathlib import Path

from lrx_cli.authenticators import create_authenticators
from lrx_cli.cache import CacheEngine
from lrx_cli.config import AppConfig, load_config
from lrx_cli.fetchers import BaseFetcher, FetcherMethodType, create_fetchers
from lrx_cli.models import TrackMeta

SAMPLE_TRACK = TrackMeta(
    title="One Last Kiss",
    artist="Hikaru Utada",
    album="One Last Kiss",
    length=252026,
    trackid="5RhWszHMSKzb7KiXk4Ae0M",
    url="https://open.spotify.com/track/5RhWszHMSKzb7KiXk4Ae0M",
)

# Sources that reach out over the network end-to-end; "local" and "cache-search"
# have no I/O worth timing.
METHODS: list[FetcherMethodType] = [
    "lrclib",
    "lrclib-search",
    "spotify",
    "musixmatch-spotify",
    "musixmatch",
    "netease",
    "qqmusic",
]

Row = tuple[str, float, str, str]


def _new_runtime(
    config: AppConfig, db_path: Path
) -> dict[FetcherMethodType, BaseFetcher]:
    cache = CacheEngine(str(db_path))
    authenticators = create_authenticators(cache, config)
    return create_fetchers(cache, authenticators, config)


async def _timed(name: str, fetcher: BaseFetcher) -> Row:
    start = time.perf_counter()
    try:
        result = await fetcher.fetch(SAMPLE_TRACK, bypass_cache=True)
        synced = result.synced.status.name if result.synced else "n/a"
        unsynced = result.unsynced.status.name if result.unsynced else "n/a"
    except Exception as exc:  # noqa: BLE001
        synced = unsynced = f"ERR: {exc}"
    elapsed_ms = (time.perf_counter() - start) * 1000
    return name, elapsed_ms, synced, unsynced


def _print_table(rows: list[Row]) -> None:
    name_w = max(max(len(name) for name, _, _, _ in rows), len("source"))
    synced_w = max(max(len(s) for _, _, s, _ in rows), len("synced"))
    unsynced_w = max(max(len(u) for _, _, _, u in rows), len("unsynced"))

    print(
        f"{'source':<{name_w}}  {'time(ms)':>10}  "
        f"{'synced':<{synced_w}}  {'unsynced':<{unsynced_w}}"
    )
    print(
        "-" * name_w + "  " + "-" * 10 + "  " + "-" * synced_w + "  " + "-" * unsynced_w
    )
    for name, elapsed_ms, synced, unsynced in rows:
        print(
            f"{name:<{name_w}}  {elapsed_ms:>10.1f}  "
            f"{synced:<{synced_w}}  {unsynced:<{unsynced_w}}"
        )


async def run_bench() -> list[Row]:
    """Time each fetcher's full `fetch()` pipeline (search/match + retrieval +
    parsing), bypassing the on-disk cache but nothing else."""
    with tempfile.TemporaryDirectory(prefix="lrx-bench-") as tmp:
        fetchers = _new_runtime(load_config(), Path(tmp) / "cred.db")
        return [await _timed(method, fetchers[method]) for method in METHODS]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Time each source's full fetch() pipeline (search/match + retrieval + "
            "parsing)."
        )
    )
    parser.parse_args()

    rows = asyncio.run(run_bench())
    _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
