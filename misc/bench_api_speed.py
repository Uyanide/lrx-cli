from __future__ import annotations

import argparse
import asyncio
import tempfile
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from lrx_cli.authenticators import create_authenticators
from lrx_cli.cache import CacheEngine
from lrx_cli.config import AppConfig, load_config
from lrx_cli.fetchers import (
    LrclibFetcher,
    LrclibSearchFetcher,
    MusixmatchFetcher,
    MusixmatchSpotifyFetcher,
    NeteaseFetcher,
    QQMusicFetcher,
    SpotifyFetcher,
    create_fetchers,
)
from lrx_cli.models import TrackMeta

SAMPLE_TRACK = TrackMeta(
    title="One Last Kiss",
    artist="Hikaru Utada",
    album="One Last Kiss",
    length=252026,
    trackid="5RhWszHMSKzb7KiXk4Ae0M",
    url="https://open.spotify.com/track/5RhWszHMSKzb7KiXk4Ae0M",
)

Row = tuple[str, float, str]


def _new_runtime(config: AppConfig, db_path: Path):
    cache = CacheEngine(str(db_path))
    authenticators = create_authenticators(cache, config)
    return create_fetchers(cache, authenticators, config)


async def _timed(name: str, fn: Callable[[], Awaitable[Any]]) -> Row:
    start = time.perf_counter()
    try:
        result = await fn()
        status = (
            str(result.status_code) if isinstance(result, httpx.Response) else "n/a"
        )
    except Exception as exc:  # noqa: BLE001
        status = f"ERR: {exc}"
    elapsed_ms = (time.perf_counter() - start) * 1000
    return name, elapsed_ms, status


def _print_table(rows: list[Row]) -> None:
    name_w = max(len(name) for name, _, _ in rows)
    status_w = max(max(len(status) for _, _, status in rows), len("status"))

    print(f"{'call':<{name_w}}  {'time(ms)':>10}  {'status':<{status_w}}")
    print("-" * name_w + "  " + "-" * 10 + "  " + "-" * status_w)
    for name, elapsed_ms, status in rows:
        print(f"{name:<{name_w}}  {elapsed_ms:>10.1f}  {status:<{status_w}}")


async def run_bench(timeout: float) -> list[Row]:
    """Time one raw HTTP round-trip per provider endpoint, bypassing app-level
    parsing/matching/caching."""
    with tempfile.TemporaryDirectory(prefix="lrx-bench-") as tmp:
        tmp_dir = Path(tmp)
        anon_fetchers = _new_runtime(AppConfig(), tmp_dir / "anon.db")
        cred_fetchers = _new_runtime(load_config(), tmp_dir / "cred.db")

        async with httpx.AsyncClient(timeout=timeout) as client:
            lrclib = anon_fetchers["lrclib"]
            assert isinstance(lrclib, LrclibFetcher)
            lrclib_search = anon_fetchers["lrclib-search"]
            assert isinstance(lrclib_search, LrclibSearchFetcher)
            netease = anon_fetchers["netease"]
            assert isinstance(netease, NeteaseFetcher)
            spotify = cred_fetchers["spotify"]
            assert isinstance(spotify, SpotifyFetcher)
            qq = cred_fetchers["qqmusic"]
            assert isinstance(qq, QQMusicFetcher)
            mxm_anon = anon_fetchers["musixmatch"]
            mxm_sp_anon = anon_fetchers["musixmatch-spotify"]
            assert isinstance(mxm_anon, MusixmatchFetcher)
            assert isinstance(mxm_sp_anon, MusixmatchSpotifyFetcher)
            mxm_cred = cred_fetchers["musixmatch"]
            mxm_sp_cred = cred_fetchers["musixmatch-spotify"]
            assert isinstance(mxm_cred, MusixmatchFetcher)
            assert isinstance(mxm_sp_cred, MusixmatchSpotifyFetcher)

            calls: list[tuple[str, Callable[[], Awaitable[Any]]]] = [
                ("lrclib_get", lambda: lrclib._api_get(client, SAMPLE_TRACK)),
                (
                    "lrclib_search_candidates",
                    lambda: lrclib_search._api_candidates(client, SAMPLE_TRACK),
                ),
                (
                    "netease_search_track",
                    lambda: netease._api_search_track(client, SAMPLE_TRACK, 5),
                ),
                (
                    "netease_lyric_track",
                    lambda: netease._api_lyric_track(client, SAMPLE_TRACK, 5),
                ),
                ("spotify_lyrics", lambda: spotify._api_lyrics(SAMPLE_TRACK)),
                ("qqmusic_search_track", lambda: qq._api_search(SAMPLE_TRACK, 10)),
                ("qqmusic_lyric_track", lambda: qq._api_lyric_track(SAMPLE_TRACK, 10)),
                (
                    "musixmatch_anonymous_search_track",
                    lambda: mxm_anon._api_search_track(SAMPLE_TRACK),
                ),
                (
                    "musixmatch_anonymous_macro_track",
                    lambda: mxm_anon._api_macro_track(SAMPLE_TRACK),
                ),
                (
                    "musixmatch_spotify_anonymous_macro_track",
                    lambda: mxm_sp_anon._api_macro_track(SAMPLE_TRACK),
                ),
                (
                    "musixmatch_token_search_track",
                    lambda: mxm_cred._api_search_track(SAMPLE_TRACK),
                ),
                (
                    "musixmatch_token_macro_track",
                    lambda: mxm_cred._api_macro_track(SAMPLE_TRACK),
                ),
                (
                    "musixmatch_spotify_token_macro_track",
                    lambda: mxm_sp_cred._api_macro_track(SAMPLE_TRACK),
                ),
            ]

            return [await _timed(name, fn) for name, fn in calls]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Time one raw HTTP round-trip per provider endpoint.")
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds.",
    )
    args = parser.parse_args()

    rows = asyncio.run(run_bench(args.timeout))
    _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
