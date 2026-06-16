"""Tweet sentiment analysis: tweet-cleaning utilities and a fine-tuning pipeline.

The lightweight preprocessing API is re-exported here for convenience
(``from src import clean_tweet_text``). The training module is intentionally not
imported eagerly, so importing this package does not pull in the heavy ML stack
(torch / transformers); use ``from src.training import ...`` when you need it.
"""

from .preprocessing import (
    clean_tweet_text,
    handle_emojis,
    normalize_hashtags,
    remove_mentions,
    remove_urls,
    to_lowercase,
)

__all__ = [
    "clean_tweet_text",
    "handle_emojis",
    "normalize_hashtags",
    "remove_mentions",
    "remove_urls",
    "to_lowercase",
]
