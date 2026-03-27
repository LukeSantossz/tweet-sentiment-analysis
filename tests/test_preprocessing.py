from src.preprocessing import (
    remove_urls,
    remove_mentions,
    normalize_hashtags,
    to_lowercase,
    handle_emojis,
    clean_tweet_text,
)


def test_remove_urls():
    tweet_data = "Aqui está o link https://site.com para ver"
    output_data = "Aqui está o link [URL] para ver"
    result_info = remove_urls(tweet_data)

    assert result_info == output_data


def test_remove_urls_border():
    tweet_data = "https://site.com"
    output_data = "[URL]"
    result_info = remove_urls(tweet_data)

    assert result_info == output_data


def test_remove_mentions():
    mention_data = "E aí @joao, beleza?"
    output_data = "E aí @user, beleza?"
    result_info = remove_mentions(mention_data)

    assert result_info == output_data


def test_remove_mult_mentions():
    mention_data = "Feliz ano novo @ana e @carlos!"
    output_data = "Feliz ano novo @user e @user!"
    result_info = remove_mentions(mention_data)

    assert result_info == output_data


def test_normalize_hashtags():
    hashtag_data = "Adoro programar em #python"
    output_data = "Adoro programar em python"
    result_info = normalize_hashtags(hashtag_data)

    assert result_info == output_data


def test_normalize_hashtags_num():
    hashtag_data = "Adoro programar em #python3"
    output_data = "Adoro programar em python3"
    result_info = normalize_hashtags(hashtag_data)

    assert result_info == output_data


def test_to_lowercase():
    lowercase_data = "Olá Mundo"
    output_data = "olá mundo"
    result_info = to_lowercase(lowercase_data)

    assert result_info == output_data


def test_to_lowercase_border():
    lowercase_data = "OLÁ MUNDO"
    output_data = "olá mundo"
    result_info = to_lowercase(lowercase_data)

    assert result_info == output_data


def test_handle_emojis():
    emoji_data = "Estou feliz 😊"
    output_data = "Estou feliz :smiling_face_with_smiling_eyes:"
    result_info = handle_emojis(emoji_data)

    assert result_info == output_data


def test_handle_emojis_border():
    emoji_data = "😊"
    output_data = ":smiling_face_with_smiling_eyes:"
    result_info = handle_emojis(emoji_data)

    assert result_info == output_data


def test_clean_tweet_text():
    tweet_data = "E aí @joao, beleza? Adoro programar em #python 😊 https://site.com"
    output_data = "e aí @user, beleza? adoro programar em python :smiling_face_with_smiling_eyes: [URL]"
    result_info = clean_tweet_text(tweet_data)

    assert result_info == output_data


def test_clean_tweet_text_border():
    tweet_data = "  E aí @joao, beleza? Adoro programar em #python 😊 https://site.com"
    output_data = "e aí @user, beleza? adoro programar em python :smiling_face_with_smiling_eyes: [URL]"
    result_info = clean_tweet_text(tweet_data)

    assert result_info == output_data
