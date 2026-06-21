"""Xquik tweet-search helpers for sentiment-analysis datasets."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.preprocessing import clean_tweet_text

DEFAULT_XQUIK_BASE_URL = "https://xquik.com/api/v1"
DEFAULT_QUERY_TYPE = "Latest"
DEFAULT_PAGE_LIMIT = 100
DEFAULT_TIMEOUT_SECONDS = 30

CSV_FIELDNAMES = (
    "tweet_id",
    "text",
    "clean_text",
    "created_at",
    "author_username",
    "author_name",
    "like_count",
    "retweet_count",
    "reply_count",
    "quote_count",
    "url",
)


@dataclass(frozen=True)
class XquikTweetRow:
    """A tweet row ready for CSV export and sentiment preprocessing."""

    tweet_id: str
    text: str
    clean_text: str
    created_at: str
    author_username: str
    author_name: str
    like_count: int
    retweet_count: int
    reply_count: int
    quote_count: int
    url: str

    def as_csv_row(self) -> dict[str, str | int]:
        """Return the row shape used by the CSV writer."""
        return {
            "tweet_id": self.tweet_id,
            "text": self.text,
            "clean_text": self.clean_text,
            "created_at": self.created_at,
            "author_username": self.author_username,
            "author_name": self.author_name,
            "like_count": self.like_count,
            "retweet_count": self.retweet_count,
            "reply_count": self.reply_count,
            "quote_count": self.quote_count,
            "url": self.url,
        }


@dataclass(frozen=True)
class XquikSearchPage:
    """One Xquik tweet-search page plus cursor metadata."""

    rows: list[XquikTweetRow]
    has_next_page: bool
    next_cursor: str | None


def build_xquik_search_url(
    query: str,
    *,
    base_url: str = DEFAULT_XQUIK_BASE_URL,
    cursor: str | None = None,
    limit: int = DEFAULT_PAGE_LIMIT,
    query_type: str = DEFAULT_QUERY_TYPE,
) -> str:
    """Build a tweet-search URL with an opaque optional cursor."""
    if query == "":
        raise ValueError("query must not be empty")
    if limit <= 0:
        raise ValueError("limit must be positive")

    params = {
        "q": query,
        "queryType": query_type,
        "limit": str(limit),
    }
    if cursor:
        params["cursor"] = cursor
    return f"{base_url.rstrip('/')}/x/tweets/search?{urlencode(params)}"


def fetch_xquik_search_page(
    api_key: str,
    query: str,
    *,
    base_url: str = DEFAULT_XQUIK_BASE_URL,
    cursor: str | None = None,
    limit: int = DEFAULT_PAGE_LIMIT,
    opener: Callable[..., Any] = urlopen,
    query_type: str = DEFAULT_QUERY_TYPE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> XquikSearchPage:
    """Fetch one search page and normalize it into dataset rows."""
    if api_key == "":
        raise ValueError("api_key must not be empty")

    request = Request(
        build_xquik_search_url(
            query,
            base_url=base_url,
            cursor=cursor,
            limit=limit,
            query_type=query_type,
        ),
        headers={"Accept": "application/json", "x-api-key": api_key},
    )
    with opener(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return parse_xquik_search_page(payload)


def fetch_xquik_search_rows(
    api_key: str,
    query: str,
    *,
    base_url: str = DEFAULT_XQUIK_BASE_URL,
    limit: int = DEFAULT_PAGE_LIMIT,
    max_pages: int = 5,
    opener: Callable[..., Any] = urlopen,
    query_type: str = DEFAULT_QUERY_TYPE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[XquikTweetRow]:
    """Fetch up to ``max_pages`` of search results."""
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")

    rows: list[XquikTweetRow] = []
    cursor: str | None = None
    for _ in range(max_pages):
        page = fetch_xquik_search_page(
            api_key,
            query,
            base_url=base_url,
            cursor=cursor,
            limit=limit,
            opener=opener,
            query_type=query_type,
            timeout_seconds=timeout_seconds,
        )
        rows.extend(page.rows)
        if not page.has_next_page or page.next_cursor is None:
            break
        cursor = page.next_cursor
    return rows


def parse_xquik_search_page(payload: Mapping[str, Any]) -> XquikSearchPage:
    """Parse an API response into normalized rows and cursor metadata."""
    tweets = payload.get("tweets", [])
    if not isinstance(tweets, Sequence) or isinstance(tweets, (str, bytes)):
        tweets = []

    rows: list[XquikTweetRow] = []
    for tweet in tweets:
        if isinstance(tweet, Mapping):
            rows.append(_tweet_to_row(tweet))

    return XquikSearchPage(
        rows=rows,
        has_next_page=bool(payload.get("has_next_page")),
        next_cursor=_optional_string(payload.get("next_cursor")),
    )


def write_xquik_rows_csv(rows: Sequence[XquikTweetRow], output_path: str | Path) -> None:
    """Write normalized tweet rows to a CSV file."""
    path = Path(output_path)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())


def _tweet_to_row(tweet: Mapping[str, Any]) -> XquikTweetRow:
    tweet_id = _string_value(tweet, ("id", "tweet_id", "rest_id"))
    text = _string_value(tweet, ("text", "full_text"))
    author = _mapping_value(tweet.get("author"))
    metrics = _mapping_value(tweet.get("metrics") or tweet.get("public_metrics"))
    username = _string_value(author, ("username", "screenName", "screen_name"))
    url = _string_value(tweet, ("url", "tweet_url"))

    return XquikTweetRow(
        tweet_id=tweet_id,
        text=text,
        clean_text=clean_tweet_text(text),
        created_at=_string_value(tweet, ("createdAt", "created_at")),
        author_username=username,
        author_name=_string_value(author, ("name", "display_name")),
        like_count=_int_value(metrics, ("like_count", "likes", "favorite_count")),
        retweet_count=_int_value(metrics, ("retweet_count", "retweets")),
        reply_count=_int_value(metrics, ("reply_count", "replies")),
        quote_count=_int_value(metrics, ("quote_count", "quotes")),
        url=url or _tweet_url(username, tweet_id),
    )


def _mapping_value(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_value(values: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = values.get(key)
        if value is not None:
            return str(value)
    return ""


def _optional_string(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _int_value(values: Mapping[str, Any], keys: Sequence[str]) -> int:
    for key in keys:
        value = values.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return 0


def _tweet_url(username: str, tweet_id: str) -> str:
    if username == "" or tweet_id == "":
        return ""
    return f"https://x.com/{username}/status/{tweet_id}"
