import sys
import os

# This block ensures Python can find your 'src' folder to import the functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the functions we built in the previous step
from preprocessing import strip_html, normalize_unicode, cleanup_whitespace, is_english, clean_text

def test_strip_html():
    """Tests if HTML tags are successfully removed."""
    raw_text = "<p>Hello <b>World</b>!</p>"
    # Replacing tags with a space leaves extra spaces, which is fine 
    # because the cleanup_whitespace function fixes it later.
    expected = " Hello  World ! " 
    assert strip_html(raw_text) == expected

def test_normalize_unicode():
    """Tests if weird unicode characters are converted to standard text."""
    # 'ﬁ' is a special ligature character that should become 'fi'
    raw_text = "This is a ﬁle."
    expected = "This is a file."
    assert normalize_unicode(raw_text) == expected

def test_cleanup_whitespace():
    """Tests if extra spaces and newlines are cleaned up."""
    raw_text = "Hello    \n\n  World"
    expected = "Hello World"
    assert cleanup_whitespace(raw_text) == expected

def test_is_english():
    """Tests the language detection filter."""
    english_text = "This is a research paper about AI."
    spanish_text = "Este es un documento sobre inteligencia artificial."
    
    assert is_english(english_text) is True
    assert is_english(spanish_text) is False

def test_clean_text_pipeline():
    """Tests the main pipeline function that combines all steps."""
    # A messy string with HTML, unicode ligatures, and bad spacing
    raw_text = "<p>  The ﬁle is  here  </p>\n\n"
    expected = "The file is here"
    
    assert clean_text(raw_text) == expected