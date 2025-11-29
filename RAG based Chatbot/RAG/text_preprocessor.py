import re
import nltk
import streamlit as st
from symspellpy import SymSpell, Verbosity
import pkg_resources
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------
# Download NLTK Resources
# -------------------------------
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)

# -------------------------------
# Cached SymSpell loader
# -------------------------------
@st.cache_resource
def load_symspell(max_edit_distance=3, prefix_length=7):
    sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=prefix_length)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt"
    )
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    return sym_spell

# -------------------------------
# Text Preprocessor Class
# -------------------------------
class TextPreprocessor:
    """
    Text preprocessing: spell correction, lemmatization, optional stopword removal.
    Handles protected words, URLs, unwanted characters, and caching heavy resources.
    """

    def __init__(self, enable_spell_check: bool = True, remove_stopwords: bool = False, protected_words=None):
        self.enable_spell_check = enable_spell_check
        self.remove_stopwords = remove_stopwords
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.lemmatizer = WordNetLemmatizer()
        self.protected_words = set(w.lower() for w in (protected_words or []))

        # Precompile regex patterns
        self.url_pattern = re.compile(r"http\S+|www\S+")
        self.non_alpha_pattern = re.compile(r"[^a-z\s_]")
        self.multi_space_pattern = re.compile(r"\s+")

        # Load SymSpell once using Streamlit cache
        self.sym_spell = load_symspell() if enable_spell_check else None

    # ----------------------------
    # Spell correction
    # ----------------------------
    def correct_spelling(self, text: str) -> str:
        if not self.enable_spell_check or not self.sym_spell:
            return text

        corrected_words = []
        for word in text.split():
            if word.lower() in self.protected_words:
                corrected_words.append(word)
                continue
            suggestions = self.sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=3)
            corrected_words.append(suggestions[0].term if suggestions else word)
        return " ".join(corrected_words)

    # ----------------------------
    # Clean and normalize text
    # ----------------------------
    def clean_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""

        # Protect domain words with placeholders
        protected_map = {}
        for w in self.protected_words:
            placeholder = f"__PROTECTED_{w.upper()}__"
            pattern = re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE)
            text = pattern.sub(placeholder, text)
            protected_map[placeholder] = w

        # 1️⃣ Lowercase
        text = text.lower()

        #  ------------------------ Remove Whitespaces ----------------------------
        pattern = re.compile(r'\s+') 
        Without_whitespace = re.sub(pattern, ' ', text)
        # There are some instances where there is no space after '?' & ')', 
        # So I am replacing these with one space so that It will not consider two words as one token.
        text = Without_whitespace.replace('?', ' ? ').replace(')', ') ')


        #  ------------------------ reducing incorrect character repeatation ----------------------------
        # Pattern matching for all case alphabets
        Pattern_alpha = re.compile(r"([A-Za-z])\1{1,}", re.DOTALL)
        # Limiting all the  repeatation to two characters.
        Formatted_text = Pattern_alpha.sub(r"\1\1", text) 
        # Pattern matching for all the punctuations that can occur
        Pattern_Punct = re.compile(r'([.,/#!$%^&*?;:{}=_`~()+-])\1{1,}')
        # Limiting punctuations in previously formatted string to only one.
        Combined_Formatted = Pattern_Punct.sub(r'\1', Formatted_text)
        # The below statement is replacing repeatation of spaces that occur more than two times with that of one occurrence.
        text = re.sub(' {2,}',' ', Combined_Formatted)

        # 2️⃣ Remove URLs and unwanted characters
        text = self.url_pattern.sub("", text)
        text = self.non_alpha_pattern.sub(" ", text)
        text = self.multi_space_pattern.sub(" ", text).strip()

        # 3️⃣ Spell correction (skip placeholders)
        if self.enable_spell_check:
            text = self.correct_spelling(text)

        # 4️⃣ Tokenize, lemmatize, remove stopwords
        words = re.findall(r'\b\w+\b', text)
        words = [self.lemmatizer.lemmatize(w) for w in words]
        if self.remove_stopwords:
            words = [w for w in words if w not in self.stop_words]

        cleaned_text = " ".join(words)

        # 5️⃣ Restore protected words
        for placeholder, original in protected_map.items():
            cleaned_text = cleaned_text.replace(placeholder.lower(), original)

        return cleaned_text
    

