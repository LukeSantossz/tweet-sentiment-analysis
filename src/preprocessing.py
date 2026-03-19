import re
import emoji

def remove_urls(text: str) -> str:
    """Remove URLs http/https."""
    url_pattern = r'https?://\S+'
    return re.sub(url_pattern, '[URL]', text).strip()

def remove_mentions(text: str) -> str:
    """Substitui @usuario por token @user."""
    return re.sub(r'@\w+', '@user', text)

def normalize_hashtags(text: str) -> str:
    """Remove o símbolo # mantendo o texto da hashtag."""
    return re.sub(r'#(\w+)', r'\1', text)

def to_lowercase(text: str) -> str:
    """Converte texto para minúsculas."""
    return text.lower()

def handle_emojis(text: str) -> str:
    """Converte emojis em texto descritivo usando notação :nome_do_emoji:."""
    return emoji.demojize(text)
   

def clean_tweet_text(text: str) -> str:
    """Encadeia todas as etapas na ordem correta."""
    text = remove_urls(text)
    text = remove_mentions(text)
    text = normalize_hashtags(text)
    text = handle_emojis(text)
    text = to_lowercase(text)
    return text
