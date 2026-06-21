import csv
import json

import pytest

from src.xquik_data import (
    build_xquik_search_url,
    fetch_xquik_search_page,
    fetch_xquik_search_rows,
    parse_xquik_search_page,
    write_xquik_rows_csv,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class FakeOpener:
    def __init__(self, pages):
        self.pages = list(pages)
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        return FakeResponse(self.pages.pop(0))


def test_build_xquik_search_url_encodes_query_and_cursor():
    url = build_xquik_search_url(
        "python sentiment lang:en",
        base_url="https://example.test/api/v1/",
        cursor="opaque cursor",
        limit=25,
    )

    assert url == (
        "https://example.test/api/v1/x/tweets/search?"
        "q=python+sentiment+lang%3Aen&queryType=Latest&limit=25&cursor=opaque+cursor"
    )


def test_build_xquik_search_url_rejects_empty_query():
    with pytest.raises(ValueError, match="query must not be empty"):
        build_xquik_search_url("")


def test_parse_xquik_search_page_normalizes_tweet_rows():
    page = parse_xquik_search_page(
        {
            "tweets": [
                {
                    "id": 1937132925391401393,
                    "text": "Great NLP update from @team #Python https://example.test",
                    "createdAt": "2026-06-20T12:00:00Z",
                    "author": {"username": "analyst", "name": "Data Analyst"},
                    "metrics": {
                        "like_count": "12",
                        "retweet_count": 3,
                        "reply_count": None,
                        "quote_count": "bad",
                    },
                }
            ],
            "has_next_page": True,
            "next_cursor": 123,
        }
    )

    row = page.rows[0]

    assert page.has_next_page is True
    assert page.next_cursor == "123"
    assert row.tweet_id == "1937132925391401393"
    assert row.author_username == "analyst"
    assert row.clean_text == "great nlp update from @user python [url]"
    assert row.like_count == 12
    assert row.retweet_count == 3
    assert row.reply_count == 0
    assert row.quote_count == 0
    assert row.url == "https://x.com/analyst/status/1937132925391401393"


def test_fetch_xquik_search_page_sends_api_key_header():
    opener = FakeOpener(
        [
            {
                "tweets": [{"id": "1", "text": "Hello #World", "author": {"username": "demo"}}],
                "has_next_page": False,
            }
        ]
    )

    page = fetch_xquik_search_page(
        "test-api-key",
        "python",
        base_url="https://example.test/api/v1",
        opener=opener,
        timeout_seconds=7,
    )

    request, timeout = opener.requests[0]

    assert timeout == 7
    assert request.headers["X-api-key"] == "test-api-key"
    assert request.headers["Accept"] == "application/json"
    assert request.full_url == "https://example.test/api/v1/x/tweets/search?q=python&queryType=Latest&limit=100"
    assert page.rows[0].clean_text == "hello world"


def test_fetch_xquik_search_rows_paginates_until_final_page():
    opener = FakeOpener(
        [
            {
                "tweets": [{"id": "1", "text": "first"}],
                "has_next_page": True,
                "next_cursor": "next-page",
            },
            {
                "tweets": [{"id": "2", "text": "second"}],
                "has_next_page": False,
            },
        ]
    )

    rows = fetch_xquik_search_rows(
        "test-api-key",
        "python",
        base_url="https://example.test/api/v1",
        opener=opener,
        limit=1,
        max_pages=3,
    )

    assert [row.tweet_id for row in rows] == ["1", "2"]
    assert opener.requests[1][0].full_url.endswith("&cursor=next-page")


def test_write_xquik_rows_csv(tmp_path):
    page = parse_xquik_search_page(
        {
            "tweets": [
                {
                    "id": "1",
                    "text": "Hello #World",
                    "author": {"username": "demo"},
                    "metrics": {"likes": 4},
                }
            ]
        }
    )
    output_path = tmp_path / "tweets.csv"

    write_xquik_rows_csv(page.rows, output_path)

    with output_path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))

    assert rows == [
        {
            "tweet_id": "1",
            "text": "Hello #World",
            "clean_text": "hello world",
            "created_at": "",
            "author_username": "demo",
            "author_name": "",
            "like_count": "4",
            "retweet_count": "0",
            "reply_count": "0",
            "quote_count": "0",
            "url": "https://x.com/demo/status/1",
        }
    ]
