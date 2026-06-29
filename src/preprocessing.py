import re

import emoji

# Rules
url_pattern = re.compile(r"https?://\S+")
mention_pattern = re.compile(r"@\w+")
hashtag_pattern = re.compile(r"#(\w+)")


def remove_urls(text: str) -> str:
    """Remove URLs http/https."""
    return re.sub(url_pattern, "[URL]", text).strip()


def remove_mentions(text: str) -> str:
    """Replace @username with the @user token."""
    return re.sub(mention_pattern, "@user", text)


def normalize_hashtags(text: str) -> str:
    """Remove the # symbol while keeping the hashtag text."""
    return re.sub(hashtag_pattern, r"\1", text)


def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def handle_emojis(text: str) -> str:
    """Convert emojis to descriptive text using the :emoji_name: notation."""
    return emoji.demojize(text)


def clean_tweet_text(text: str) -> str:
    """Chain all cleaning steps in the correct order."""
    text = remove_urls(text)
    text = remove_mentions(text)
    text = normalize_hashtags(text)
    text = handle_emojis(text)
    text = to_lowercase(text)
    return text


def preprocess_for_model(text: str) -> str:
    """Normalize text to the CardiffNLP twitter-roberta-base-sentiment convention.

    Mirrors the model's official preprocessing: collapse @mentions to ``@user`` and
    URLs to ``http``, while preserving case, hashtags, and raw emoji. This is the
    single contract shared by the training tokenizer and the future serving path
    (issue #36); the generic ``clean_tweet_text`` is intentionally not used on the
    model path (see docs/adr/0009).
    """
    tokens = []
    for token in text.split(" "):
        if token.startswith("@") and len(token) > 1:
            token = "@user"
        elif token.startswith("http"):
            token = "http"
        tokens.append(token)
    return " ".join(tokens)
