"""Tests for TokenizerUtils functionality."""

import pytest
from unittest.mock import Mock
from src.llama_lora.utils.common import TokenizerUtils


class TestTokenizerUtils:
    """Test cases for TokenizerUtils class."""
    
    def test_setup_tokenizer_with_no_pad_token(self):
        """Test tokenizer setup when pad_token is None."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<EOS>"
        
        result = TokenizerUtils.setup_tokenizer(mock_tokenizer)
        
        assert result.pad_token == "<EOS>"
        assert result == mock_tokenizer
    
    def test_setup_tokenizer_with_existing_pad_token(self):
        """Test tokenizer setup when pad_token already exists."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<PAD>"
        mock_tokenizer.eos_token = "<EOS>"
        
        result = TokenizerUtils.setup_tokenizer(mock_tokenizer)
        
        assert result.pad_token == "<PAD>"
        assert result == mock_tokenizer

    def test_format_alpaca_prompt_complete_data(self):
        """Test Alpaca prompt formatting with complete data."""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}
        
        example = {
            "instruction": "Translate this text",
            "input": "Hello world",
            "output": "Hola mundo"
        }
        
        result = TokenizerUtils.format_alpaca_prompt(example, mock_tokenizer, 512)
        
        expected_text = (
            "### Instruction:\nTranslate this text\n"
            "### Input:\nHello world\n"
            "### Response:\nHola mundo"
        )
        mock_tokenizer.assert_called_once_with(
            expected_text, truncation=True, max_length=512, padding="max_length"
        )
        assert result == {"input_ids": [1, 2, 3]}

    def test_format_alpaca_prompt_missing_fields(self):
        """Test Alpaca prompt formatting with missing fields."""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}
        
        # Test with missing instruction
        example = {"input": "Hello", "output": "Hi"}
        result = TokenizerUtils.format_alpaca_prompt(example, mock_tokenizer, 512)
        
        # Should use default instruction
        expected_text = (
            "### Instruction:\nPlease respond to the following.\n"
            "### Input:\nHello\n"
            "### Response:\nHi"
        )
        mock_tokenizer.assert_called_with(
            expected_text, truncation=True, max_length=512, padding="max_length"
        )

    def test_format_alpaca_prompt_empty_input(self):
        """Test Alpaca prompt formatting with empty input field."""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}
        
        example = {
            "instruction": "Say hello",
            "input": "",
            "output": "Hello!"
        }
        
        result = TokenizerUtils.format_alpaca_prompt(example, mock_tokenizer, 512)
        
        expected_text = (
            "### Instruction:\nSay hello\n"
            "### Input:\n\n"
            "### Response:\nHello!"
        )
        mock_tokenizer.assert_called_once_with(
            expected_text, truncation=True, max_length=512, padding="max_length"
        )

    def test_format_alpaca_prompt_missing_output(self):
        """Test Alpaca prompt formatting with missing output field."""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}
        
        example = {
            "instruction": "Say hello",
            "input": "user input"
        }
        
        result = TokenizerUtils.format_alpaca_prompt(example, mock_tokenizer, 512)
        
        expected_text = (
            "### Instruction:\nSay hello\n"
            "### Input:\nuser input\n"
            "### Response:\nI understand."
        )
        mock_tokenizer.assert_called_once_with(
            expected_text, truncation=True, max_length=512, padding="max_length"
        )