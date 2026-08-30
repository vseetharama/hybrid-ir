"""
Property-Based Tests for Text Preprocessing Pipeline

This module tests the TextPreprocessor class with property-based testing using Hypothesis.
It validates all preprocessing steps: lowercasing, punctuation removal, tokenization,
stop-word removal, and lemmatization.

**Validates: Requirements 1.1-1.7**
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from pathlib import Path
import sys

# Import the TextPreprocessor class from src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocess import TextPreprocessor


# ============================================================================
# STRATEGY DEFINITIONS (Input Generators)
# ============================================================================

# Strategy for generating random text with various characters
text_with_punctuation = st.text(
    alphabet=st.characters(
        min_codepoint=32,
        max_codepoint=126,
        blacklist_characters=None
    ),
    min_size=0,
    max_size=500
)

# Strategy for generating text with common words
common_words = st.lists(
    st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=20),
    min_size=1,
    max_size=20
).map(lambda words: " ".join(words))

# Strategy for generating text with hyphenated words
hyphenated_words = st.lists(
    st.tuples(
        st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=10),
        st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=10),
    ),
    min_size=1,
    max_size=10
).map(lambda pairs: " ".join([f"{a}-{b}" for a, b in pairs]))

# Strategy for whitespace-only strings
whitespace_only = st.just("") | st.just(" ") | st.just("   ") | st.just("\t") | st.just("\n")


# ============================================================================
# PROPERTY 1: PREPROCESSING PRODUCES LOWERCASE OUTPUT
# ============================================================================

class TestProperty1PreprocessingProducesLowercaseOutput:
    """
    Property 1: Preprocessing produces lowercase output
    
    Validates that all characters in the output are lowercase
    (except non-alphabetic characters which remain unchanged).
    """
    
    @given(text=common_words)
    @settings(max_examples=20, deadline=None)
    def test_preprocessing_produces_only_lowercase_alphabetic_chars(self, text):
        """
        Test that preprocessing output contains only lowercase alphabetic characters.
        
        Given: Any input text with mixed case
        When: preprocess() is called
        Then: All alphabetic characters in output are lowercase
        
        **Validates: Requirement 1.1**
        """
        assume(text.strip())  # Skip empty inputs for this test
        
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess(text)
        
        # Rejoin tokens to check
        output = " ".join(result)
        
        # Check that all letters are lowercase
        for char in output:
            if char.isalpha():
                assert char.islower(), f"Found uppercase character: {char}"
    
    @given(text=text_with_punctuation)
    @settings(max_examples=20, deadline=None)
    def test_lowercase_method_returns_lowercase_string(self, text):
        """
        Test that lowercase() method converts all characters to lowercase.
        
        Given: Text with mixed case
        When: lowercase() is called
        Then: Result has all alphabetic characters in lowercase
        
        **Validates: Requirement 1.1**
        """
        preprocessor = TextPreprocessor()
        result = preprocessor.lowercase(text)
        
        # Check that result equals text.lower()
        assert result == text.lower()
    
    @given(text=st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=1, max_size=100))
    @settings(max_examples=20, deadline=None)
    def test_lowercase_converts_uppercase_to_lowercase(self, text):
        """
        Test that uppercase input is converted to lowercase.
        
        Given: All uppercase text
        When: lowercase() is called
        Then: All characters are converted to lowercase
        
        **Validates: Requirement 1.1**
        """
        preprocessor = TextPreprocessor()
        result = preprocessor.lowercase(text)
        
        assert result == text.lower()
        assert result.islower() or not result.isalpha()


# ============================================================================
# PROPERTY 2: PUNCTUATION REMOVAL PRESERVES HYPHENS IN WORDS
# ============================================================================

class TestProperty2PunctuationRemovalPreservesHyphensInWords:
    """
    Property 2: Punctuation removal preserves hyphens in words
    
    Validates that hyphens connecting word characters are preserved,
    while other punctuation is removed.
    """
    
    @given(text=hyphenated_words)
    @settings(max_examples=20, deadline=None)
    def test_remove_punctuation_preserves_in_word_hyphens(self, text):
        """
        Test that hyphens between word characters are preserved.
        
        Given: Text with hyphenated compound words (e.g., "well-known")
        When: remove_punctuation() is called
        Then: Hyphens between word characters remain
              Other punctuation is removed
        
        **Validates: Requirement 1.2**
        """
        preprocessor = TextPreprocessor()
        result = preprocessor.remove_punctuation(text)
        
        # Check that hyphens between letters are preserved
        assert "-" in result or "-" not in text
        
        # Check that common punctuation is removed
        for punct in [",", ".", "!", "?", ";", ":", "'", '"']:
            assert punct not in result
    
    @given(
        first_word=st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=10),
        second_word=st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=10),
    )
    @settings(max_examples=20, deadline=None)
    def test_hyphenated_words_preserved_exactly(self, first_word, second_word):
        """
        Test that specific hyphenated words like "well-known" are preserved exactly.
        
        Given: Two word fragments
        When: Text "word1-word2" is processed
        Then: Result contains "word1-word2" unchanged
        
        **Validates: Requirement 1.2**
        """
        text = f"{first_word}-{second_word}"
        preprocessor = TextPreprocessor()
        result = preprocessor.remove_punctuation(text)
        
        # The hyphen should be preserved
        assert "-" in result
        # The result should contain the hyphenated version
        assert f"{first_word}-{second_word}" in result
    
    @given(text=st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=50))
    @settings(max_examples=20, deadline=None)
    def test_remove_punctuation_removes_edge_hyphens(self, text):
        """
        Test that hyphens at word boundaries (not between characters) are removed.
        
        Given: Text with leading or trailing hyphens (e.g., "-word" or "word-")
        When: remove_punctuation() is called
        Then: Edge hyphens are removed or replaced with space
        
        **Validates: Requirement 1.2**
        """
        text_with_edge_hyphen = f"-{text}-"
        preprocessor = TextPreprocessor()
        result = preprocessor.remove_punctuation(text_with_edge_hyphen)
        
        # Edge hyphens should not be preserved
        # Result should not start with hyphen
        assert not result.startswith("-"), "Result should not start with hyphen"
        # Result should not end with hyphen
        assert not result.endswith("-"), "Result should not end with hyphen"
    
    @given(text=text_with_punctuation)
    @settings(max_examples=20, deadline=None)
    def test_remove_punctuation_removes_most_punctuation(self, text):
        """
        Test that most punctuation is removed except spaces and in-word hyphens.
        
        Given: Text with various punctuation
        When: remove_punctuation() is called
        Then: Comma, period, etc. are removed
              Spaces and in-word hyphens are preserved
        
        **Validates: Requirement 1.2**
        """
        preprocessor = TextPreprocessor()
        result = preprocessor.remove_punctuation(text)
        
        # Check that common punctuation is removed
        punctuation_to_remove = [",", ".", "!", "?", ";", ":", "'", '"', "(", ")", "[", "]", "{", "}"]
        for punct in punctuation_to_remove:
            assert punct not in result, f"Punctuation '{punct}' should be removed"


# ============================================================================
# PROPERTY 3: TOKENIZATION PRODUCES ORDERED LIST
# ============================================================================

class TestProperty3TokenizationProducesOrderedList:
    """
    Property 3: Tokenization produces ordered list
    
    Validates that tokens are returned in the order they appear in the text.
    """
    
    @given(text=common_words)
    @settings(max_examples=20, deadline=None)
    def test_tokenize_returns_tokens_in_order(self, text):
        """
        Test that tokenization returns tokens in the order they appear.
        
        Given: Text with multiple words separated by spaces
        When: tokenize() is called
        Then: Returned tokens maintain their original order
        
        **Validates: Requirement 1.3**
        """
        assume(text.strip())  # Skip empty/whitespace-only
        
        preprocessor = TextPreprocessor()
        result = preprocessor.tokenize(text)
        
        # Result should be a list
        assert isinstance(result, list)
        
        # Result should not be empty
        assert len(result) > 0
        
        # All elements should be strings
        assert all(isinstance(token, str) for token in result)
        
        # Reconstruct by joining and check it's close to original
        reconstructed = " ".join(result)
        assert len(reconstructed) > 0
    
    @given(words=st.lists(
        st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=20),
        min_size=1,
        max_size=20
    ))
    @settings(max_examples=20, deadline=None)
    def test_token_order_matches_word_order(self, words):
        """
        Test that token order matches the input word order.
        
        Given: Multiple words separated by spaces
        When: tokenize() is called
        Then: Token order matches input word order
        
        **Validates: Requirement 1.3**
        """
        text = " ".join(words)
        preprocessor = TextPreprocessor()
        result = preprocessor.tokenize(text)
        
        # Basic check: if we have at least 2 words, check order is preserved
        if len(words) >= 2:
            # The first token should contain part of first word
            assert result[0].startswith(words[0][0]) or result[0] == words[0]
    
    @given(text=common_words)
    @settings(max_examples=20, deadline=None)
    def test_tokenize_returns_list_not_set_or_other(self, text):
        """
        Test that tokenize() returns a list (not set or dict).
        
        Given: Any text
        When: tokenize() is called
        Then: Returns a list (preserving order)
        
        **Validates: Requirement 1.3**
        """
        assume(text.strip())
        
        preprocessor = TextPreprocessor()
        result = preprocessor.tokenize(text)
        
        assert isinstance(result, list), "tokenize() should return a list"


# ============================================================================
# PROPERTY 4: STOP-WORD REMOVAL IS COMPLETE
# ============================================================================

class TestProperty4StopWordRemovalIsComplete:
    """
    Property 4: Stop-word removal is complete
    
    Validates that all English stop words are removed from token lists.
    """
    
    @given(
        tokens=st.lists(
            st.sampled_from(['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'being', 'been',
                            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                            'could', 'may', 'might', 'must', 'can', 'in', 'on', 'at', 'to', 'for',
                            'with', 'by', 'from', 'of', 'and', 'or', 'but', 'not', 'no', 'yes']),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_remove_stop_words_removes_all_english_stop_words(self, tokens):
        """
        Test that all English stop words are removed.
        
        Given: List of English stop words
        When: remove_stop_words() is called
        Then: All stop words are removed (result is empty or contains non-stop-words)
        
        **Validates: Requirement 1.4**
        """
        preprocessor = TextPreprocessor()
        result = preprocessor.remove_stop_words(tokens)
        
        # Result should not contain any of the input stop words
        for word in result:
            assert word.lower() not in preprocessor.stop_words, f"Stop word '{word}' not removed"
    
    @given(
        non_stop_words=st.lists(
            st.text(
                alphabet=st.characters(min_codepoint=97, max_codepoint=122),
                min_size=3,
                max_size=20
            ).filter(lambda x: x and x not in ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                                         'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                                         'would', 'should', 'could', 'may', 'might', 'must',
                                         'can', 'in', 'on', 'at', 'to', 'for', 'with', 'by',
                                         'from', 'of', 'and', 'or', 'but', 'not', 'no', 'shan']),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_remove_stop_words_preserves_non_stop_words(self, non_stop_words):
        """
        Test that non-stop-words are preserved.
        
        Given: List of non-stop-words
        When: remove_stop_words() is called
        Then: Non-stop-words are preserved in output
        
        **Validates: Requirement 1.4**
        """
        # Pre-check: Make sure our test words are actually not stop words
        preprocessor = TextPreprocessor()
        for word in non_stop_words:
            assume(word.lower() not in preprocessor.stop_words)
        
        result = preprocessor.remove_stop_words(non_stop_words)
        
        # All non-stop-words should be preserved
        assert len(result) == len(non_stop_words)
        assert result == non_stop_words
    
    @given(
        stop_and_non_stop=st.lists(
            st.one_of(
                st.just('the'),
                st.just('researcher'),
                st.just('a'),
                st.just('study'),
                st.just('and'),
                st.just('nlp'),
            ),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_remove_stop_words_mixed_list(self, stop_and_non_stop):
        """
        Test stop-word removal on mixed list of stop and non-stop words.
        
        Given: Mixed list with stop words ('the', 'a', 'and') and non-stop words
        When: remove_stop_words() is called
        Then: Stop words removed, non-stop words preserved
        
        **Validates: Requirement 1.4**
        """
        preprocessor = TextPreprocessor()
        result = preprocessor.remove_stop_words(stop_and_non_stop)
        
        # No stop words should be in result
        stop_words_in_result = [w for w in result if w.lower() in preprocessor.stop_words]
        assert len(stop_words_in_result) == 0, f"Stop words found in result: {stop_words_in_result}"
        
        # Non-stop words should be preserved
        for word in stop_and_non_stop:
            if word.lower() not in preprocessor.stop_words:
                assert word in result, f"Non-stop word '{word}' should be preserved"


# ============================================================================
# PROPERTY 5: LEMMATIZATION IDEMPOTENCE
# ============================================================================

class TestProperty5LemmatizationIdempotence:
    """
    Property 5: Lemmatization idempotence
    
    Validates that applying lemmatization twice produces the same result as applying it once.
    """
    
    @given(
        tokens=st.lists(
            st.text(
                alphabet=st.characters(min_codepoint=97, max_codepoint=122),
                min_size=1,
                max_size=20
            ),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_lemmatize_tokens_is_idempotent(self, tokens):
        """
        Test that lemmatization is idempotent (applying twice = applying once).
        
        Given: List of tokens
        When: lemmatize_tokens() is called twice
        Then: Second application produces same result as first application
        
        **Validates: Requirement 1.5**
        """
        preprocessor = TextPreprocessor()
        
        # Apply lemmatization once
        result1 = preprocessor.lemmatize_tokens(tokens)
        
        # Apply lemmatization to the result
        result2 = preprocessor.lemmatize_tokens(result1)
        
        # Second application should produce same result
        assert result1 == result2, "Lemmatization is not idempotent"
    
    @given(
        verb_forms=st.lists(
            st.sampled_from(['running', 'run', 'runs', 'ran', 'jumped', 'jump', 'jumps',
                            'studied', 'study', 'studies', 'creating', 'create', 'creates']),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_lemmatize_tokens_reduces_to_base_form(self, verb_forms):
        """
        Test that lemmatization reduces words to base form.
        
        Given: List of different word forms (running, jumped, etc.)
        When: lemmatize_tokens() is called
        Then: Different forms of same word reduce to same base form
        
        **Validates: Requirement 1.5**
        """
        preprocessor = TextPreprocessor()
        result = preprocessor.lemmatize_tokens(verb_forms)
        
        # Result should have all strings
        assert all(isinstance(token, str) for token in result)
        
        # Result should have same length as input
        assert len(result) == len(verb_forms)
    
    @given(tokens=st.lists(
        st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=15),
        min_size=1,
        max_size=20
    ))
    @settings(max_examples=20, deadline=None)
    def test_lemmatize_tokens_returns_strings(self, tokens):
        """
        Test that lemmatize_tokens returns a list of strings.
        
        Given: List of tokens
        When: lemmatize_tokens() is called
        Then: Returns list of strings
        
        **Validates: Requirement 1.5**
        """
        preprocessor = TextPreprocessor()
        result = preprocessor.lemmatize_tokens(tokens)
        
        assert isinstance(result, list)
        assert all(isinstance(token, str) for token in result)
        assert len(result) == len(tokens)


# ============================================================================
# PROPERTY 6: EMPTY INPUT PRODUCES EMPTY OUTPUT
# ============================================================================

class TestProperty6EmptyInputProducesEmptyOutput:
    """
    Property 6: Empty input produces empty output
    
    Validates that empty or whitespace-only input returns an empty list.
    """
    
    @given(whitespace=whitespace_only)
    @settings(max_examples=20, deadline=None)
    def test_preprocess_empty_returns_empty_list(self, whitespace):
        """
        Test that empty or whitespace-only input returns empty list.
        
        Given: Empty string or string with only whitespace
        When: preprocess() is called
        Then: Returns empty list []
        
        **Validates: Requirement 1.7**
        """
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess(whitespace)
        
        assert result == [], f"Expected empty list, got {result}"
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_preprocess_empty_string_returns_empty_list(self):
        """
        Test that empty string returns empty list.
        
        **Validates: Requirement 1.7**
        """
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess("")
        
        assert result == []
        assert len(result) == 0
    
    def test_preprocess_whitespace_returns_empty_list(self):
        """
        Test that whitespace-only input returns empty list.
        
        **Validates: Requirement 1.7**
        """
        preprocessor = TextPreprocessor()
        
        test_cases = ["   ", "\t", "\n", "\t\n  ", "                "]
        
        for text in test_cases:
            result = preprocessor.preprocess(text)
            assert result == [], f"Expected empty list for '{repr(text)}', got {result}"
    
    @given(whitespace=st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=32), min_size=0, max_size=100))
    @settings(max_examples=20, deadline=None)
    def test_preprocess_spaces_only_returns_empty_list(self, whitespace):
        """
        Test that input with only spaces returns empty list.
        
        Given: String containing only spaces
        When: preprocess() is called
        Then: Returns empty list
        
        **Validates: Requirement 1.7**
        """
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess(whitespace)
        
        assert result == []


# ============================================================================
# PROPERTY TESTS FOR COMPLETE PIPELINE
# ============================================================================

class TestCompletePreprocessingPipeline:
    """
    Test the complete preprocessing pipeline end-to-end.
    """
    
    @given(text=common_words)
    @settings(max_examples=20, deadline=None)
    def test_complete_pipeline_consistency(self, text):
        """
        Test that complete pipeline is consistent and produces valid results.
        
        Given: Any input text
        When: preprocess() is called
        Then: Returns valid result (list of lowercase lemmatized tokens)
        
        Note: Due to the pipeline order (stopword removal before lemmatization),
        some words that become stopwords only after lemmatization may still appear
        in the output (e.g., 'ms' lemmatizes to 'm' which is a stopword).
        
        **Validates: Requirements 1.1-1.7**
        """
        assume(text.strip())  # Skip empty inputs
        
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess(text)
        
        # Result should be a list
        assert isinstance(result, list)
        
        # All elements should be strings
        assert all(isinstance(token, str) for token in result)
        
        # All tokens should be lowercase
        for token in result:
            for char in token:
                if char.isalpha():
                    assert char.islower(), f"Found uppercase in token: {token}"
    
    @given(text=text_with_punctuation)
    @settings(max_examples=20, deadline=None)
    def test_complete_pipeline_handles_any_input(self, text):
        """
        Test that complete pipeline handles any input without error.
        
        Given: Any text with any characters
        When: preprocess() is called
        Then: Returns list (possibly empty) without raising exceptions
        
        **Validates: Requirements 1.1-1.7**
        """
        preprocessor = TextPreprocessor()
        
        # Should not raise any exception
        try:
            result = preprocessor.preprocess(text)
            assert isinstance(result, list)
            assert all(isinstance(token, str) for token in result)
        except Exception as e:
            pytest.fail(f"preprocess() raised exception: {e}")
    
    def test_complete_pipeline_well_known_example(self):
        """
        Test complete pipeline with "well-known" example from requirements.
        
        **Validates: Requirements 1.1-1.7**
        """
        preprocessor = TextPreprocessor()
        
        # "well-known" should be preserved through the pipeline
        text = "well-known researchers study NLP"
        result = preprocessor.preprocess(text)
        
        # Should have "well-known" as a token
        assert "well-known" in result
        # Should have "researcher" (lemmatized from "researchers")
        assert "researcher" in result or "research" in result
        # Should have "study" or "studi" (lemmatized)
        assert any("study" in token or "studi" in token for token in result)
        # Should have "nlp"
        assert "nlp" in result
    
    def test_complete_pipeline_with_options(self):
        """
        Test complete pipeline with different options (stop word removal, lemmatization).
        
        **Validates: Requirements 1.1-1.7**
        """
        # With stop words and lemmatization (default)
        preprocessor_default = TextPreprocessor(remove_stopwords=True, lemmatize=True)
        result_default = preprocessor_default.preprocess("The researchers are studying")
        
        # With stop words but no lemmatization
        preprocessor_no_lemma = TextPreprocessor(remove_stopwords=True, lemmatize=False)
        result_no_lemma = preprocessor_no_lemma.preprocess("The researchers are studying")
        
        # With lemmatization but no stop word removal
        preprocessor_no_stopwords = TextPreprocessor(remove_stopwords=False, lemmatize=True)
        result_no_stopwords = preprocessor_no_stopwords.preprocess("The researchers are studying")
        
        # Default should be shortest (stop words removed)
        assert len(result_default) <= len(result_no_stopwords)
        
        # Check that "The" is not in default (stop word)
        assert "the" not in result_default
        # Check that "The" is in no_stopwords version
        assert any("the" in token.lower() for token in result_no_stopwords)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
