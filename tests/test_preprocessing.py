from src.preprocessing import (
    clean_tweet_text,
    handle_emojis,
    normalize_hashtags,
    remove_mentions,
    remove_urls,
    to_lowercase,
)


def test_remove_urls():
    tweet_data = "Here is the link https://site.com to see"
    output_data = "Here is the link [URL] to see"
    result_info = remove_urls(tweet_data)

    assert result_info == output_data


def test_remove_urls_border():
    tweet_data = "https://site.com"
    output_data = "[URL]"
    result_info = remove_urls(tweet_data)

    assert result_info == output_data


def test_remove_mentions():
    mention_data = "Hey @joao, all good?"
    output_data = "Hey @user, all good?"
    result_info = remove_mentions(mention_data)

    assert result_info == output_data


def test_remove_mult_mentions():
    mention_data = "Happy new year @ana and @carlos!"
    output_data = "Happy new year @user and @user!"
    result_info = remove_mentions(mention_data)

    assert result_info == output_data


def test_normalize_hashtags():
    hashtag_data = "I love programming in #python"
    output_data = "I love programming in python"
    result_info = normalize_hashtags(hashtag_data)

    assert result_info == output_data


def test_normalize_hashtags_num():
    hashtag_data = "I love programming in #python3"
    output_data = "I love programming in python3"
    result_info = normalize_hashtags(hashtag_data)

    assert result_info == output_data


def test_to_lowercase():
    lowercase_data = "Hello World"
    output_data = "hello world"
    result_info = to_lowercase(lowercase_data)

    assert result_info == output_data


def test_to_lowercase_border():
    lowercase_data = "HELLO WORLD"
    output_data = "hello world"
    result_info = to_lowercase(lowercase_data)

    assert result_info == output_data


def test_handle_emojis():
    emoji_data = "I am happy 😊"
    output_data = "I am happy :smiling_face_with_smiling_eyes:"
    result_info = handle_emojis(emoji_data)

    assert result_info == output_data


def test_handle_emojis_border():
    emoji_data = "😊"
    output_data = ":smiling_face_with_smiling_eyes:"
    result_info = handle_emojis(emoji_data)

    assert result_info == output_data


def test_clean_tweet_text():
    tweet_data = "Hey @joao, all good? I love programming in #python 😊 https://site.com"
    output_data = "hey @user, all good? i love programming in python :smiling_face_with_smiling_eyes: [url]"
    result_info = clean_tweet_text(tweet_data)

    assert result_info == output_data


def test_clean_tweet_text_border():
    tweet_data = "  Hey @joao, all good? I love programming in #python 😊 https://site.com"
    output_data = "hey @user, all good? i love programming in python :smiling_face_with_smiling_eyes: [url]"
    result_info = clean_tweet_text(tweet_data)

    assert result_info == output_data
