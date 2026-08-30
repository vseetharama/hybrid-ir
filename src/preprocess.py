"""
Text preprocessing module for the Hybrid IR System.

This module provides a TextPreprocessor class that implements a complete
text preprocessing pipeline including lowercasing, punctuation removal,
tokenization, stop-word removal, and lemmatization.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7
"""

import re
import string
from typing import List, Optional
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _download_nltk_data():
    """Ensure required NLTK data is available for explicit runtime use."""
    resources = ['punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            if 'tokenizers' in resource or resource == 'punkt_tab':
                nltk.data.find(f'tokenizers/{resource}')
            else:
                nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)


class TextPreprocessor:
    """
    Text preprocessing pipeline for documents and queries.
    
    Implements a complete preprocessing pipeline:
    1. Lowercase
    2. Remove punctuation (except spaces and hyphens in words)
    3. Tokenize using NLTK word_tokenize
    4. Remove English stop words
    5. Lemmatize each token
    
    Attributes:
        remove_stopwords (bool): Whether to remove stop words
        lemmatize (bool): Whether to apply lemmatization
        stop_words (set): Set of English stop words
        lemmatizer (WordNetLemmatizer): Lemmatizer instance
    """
    
    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        download_nltk_data: bool = True,
    ):
        """
        Initialize the TextPreprocessor.
        
        Args:
            remove_stopwords (bool): Whether to remove stop words. Defaults to True.
            lemmatize (bool): Whether to apply lemmatization. Defaults to True.
            download_nltk_data (bool): Whether missing NLTK resources may be
                downloaded. Disable this for import-safe application startup.
        """
        if download_nltk_data:
            try:
                _download_nltk_data()
            except Exception:
                # Resource access is checked below so startup can degrade safely.
                pass

        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.stop_words = set()
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess(self, text: str) -> List[str]:
        """
        Complete preprocessing pipeline.
        
        Applies the following transformations in order:
        1. Lowercase
        2. Remove punctuation (except spaces and hyphens within words)
        3. Tokenize using NLTK word_tokenize
        4. Remove stop words (if enabled)
        5. Lemmatize (if enabled)
        
        Args:
            text (str): Raw input text to preprocess
            
        Returns:
            List[str]: List of preprocessed tokens in order
            
        Requirements: 1.1-1.7
        
        Examples:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.preprocess("Hello, World!")
            ['hello', 'world']
            
            >>> preprocessor.preprocess("Well-known researchers study NLP.")
            ['well-known', 'researcher', 'study', 'nlp']
            
            >>> preprocessor.preprocess("   ")
            []
        """
        # Handle empty or whitespace-only input (Requirement 1.7)
        if not text or not text.strip():
            return []
        
        # Step 1: Lowercase (Requirement 1.1)
        text = self.lowercase(text)
        
        # Step 2: Remove punctuation (Requirement 1.2)
        text = self.remove_punctuation(text)
        
        # Step 3: Tokenize (Requirement 1.3)
        tokens = self.tokenize(text)
        
        # Step 4: Remove stop words (Requirement 1.4)
        if self.remove_stopwords:
            tokens = self.remove_stop_words(tokens)
        
        # Step 5: Lemmatize (Requirement 1.5)
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        return tokens
    
    def lowercase(self, text: str) -> str:
        """
        Convert text to lowercase.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Lowercased text
            
        Requirement: 1.1
        """
        return text.lower()
    
    def remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation except spaces and hyphens within words.
        
        This method removes all punctuation characters except:
        - Spaces (word separators)
        - Hyphens that appear within words (e.g., "well-known")
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with punctuation removed
            
        Requirement: 1.2
        
        Examples:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.remove_punctuation("Hello, world!")
            'Hello world'
            
            >>> preprocessor.remove_punctuation("well-known")
            'well-known'
            
            >>> preprocessor.remove_punctuation("test's")
            'tests'
        """
        # Keep spaces and hyphens that are between word characters
        # Replace punctuation with space to avoid word concatenation
        result = []
        for i, char in enumerate(text):
            if char in string.punctuation:
                # Keep hyphens if they are surrounded by word characters
                if char == '-':
                    # Check if hyphen is between word characters
                    if (i > 0 and i < len(text) - 1 and 
                        text[i-1].isalnum() and text[i+1].isalnum()):
                        result.append(char)
                    else:
                        # Replace edge hyphens with space
                        result.append(' ')
                else:
                    # Replace other punctuation with space
                    result.append(' ')
            else:
                result.append(char)
        
        # Clean up multiple consecutive spaces
        text = ''.join(result)
        text = ' '.join(text.split())
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using NLTK word_tokenize.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of tokens
            
        Requirement: 1.3
        """
        tokens = word_tokenize(text)
        return tokens
    
    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """
        Remove English stop words from token list.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: List of tokens with stop words removed
            
        Requirement: 1.4
        """
        filtered_tokens = [token for token in tokens if token.lower() not in self.stop_words]
        return filtered_tokens
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply lemmatization to each token.
        
        Uses WordNetLemmatizer to reduce words to their base form.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: List of lemmatized tokens
            
        Requirement: 1.5
        """
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens
