import re
from bs4 import BeautifulSoup

def clean_text(text):
    """Clean text by removing URLs and HTML tags"""
    if not text:
        return ''
    
    text = re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', str(text))
    text = BeautifulSoup(text, 'lxml').get_text()
    
    return text.strip()

def format_genres(genres):
    """Format genre list for display"""
    if isinstance(genres, list):
        return ', '.join(genres[:3])  # Show top 3 genres
    elif isinstance(genres, str):
        return genres
    return ''

def get_poster_url(poster_path, size='w200'):
    """Get full poster URL from TMDB path"""
    if poster_path and isinstance(poster_path, str) and poster_path.startswith('/'):
        return f"https://image.tmdb.org/t/p/{size}{poster_path}"
    return None