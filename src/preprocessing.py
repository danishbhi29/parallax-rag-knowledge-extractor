import re
import unicodedata
from langdetect import detect, LangDetectException

def strip_html(text: str) -> str:
    """
    Removes HTML tags from the text using Regular Expressions.
    Example: '<p>Hello</p>' becomes 'Hello'
    """
    # The regex <[^>]+> matches any text inside angle brackets
    return re.sub(r'<[^>]+>', ' ', text)

def normalize_unicode(text: str) -> str:
    """
    Normalizes unicode characters to standard form.
    Example: Converts special ligatures like 'ﬁ' into standard 'fi'.
    """
    # NFKC is a standard normalization form that fixes weird characters
    return unicodedata.normalize('NFKC', text)

def cleanup_whitespace(text: str) -> str:
    """
    Replaces multiple spaces, tabs, and newlines with a single space.
    Also removes leading and trailing whitespace.
    """
    # \s+ matches any whitespace character (spaces, tabs, newlines)
    return re.sub(r'\s+', ' ', text).strip()

def is_english(text: str) -> bool:
    """
    Checks if the text is primarily in English.
    Returns True if English, False otherwise.
    """
    try:
        # The detect() function returns 'en' if the text is English
        return detect(text) == 'en'
    except LangDetectException:
        # If the text is too short or empty, it might fail to detect
        return False

def clean_text(text: str) -> str:
    """
    Main pipeline function. Applies all cleaning steps in the correct order.
    """
    text = strip_html(text)
    text = normalize_unicode(text)
    text = cleanup_whitespace(text)
    return text