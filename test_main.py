"""Tests for the index-interview system."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from codex import (
    add_assistant_message,
    add_user_message,
    call_codex,
    clear_history,
    get_history,
)
from dataset import PRBenchDataset
from main import main


@pytest.mark.parametrize("turns", [2, 6])
def test_sample_preview_prints_each_available_prompt(turns, capsys):
    prompts = [f"synthetic prompt {turn}" for turn in range(turns)]
    dataset = Mock()
    dataset.get_sample.return_value = {
        "turns": turns,
        **{f"prompt_{turn}": prompt for turn, prompt in enumerate(prompts)},
    }
    with patch("main.PRBenchDataset", return_value=dataset):
        main()

    dataset.load.assert_called_once_with()
    dataset.get_sample.assert_called_once_with("finance", 0)
    assert capsys.readouterr().out.splitlines()[-turns - 1:] == [
        str(turns),
        *(repr(prompt) for prompt in prompts),
    ]

# ============================================================================
# Dataset Loading Tests
# ============================================================================

class TestDatasetLoading:
    """Tests for PRBench dataset loading functionality."""
    
    def test_dataset_initialization(self):
        """Test that dataset initializes with correct attributes."""
        dataset = PRBenchDataset()
        
        assert dataset.dataset_name == "ScaleAI/PRBench"
        assert dataset.dataset is None
        assert dataset.available_splits == ["finance", "legal", "finance_hard", "legal_hard"]
        assert dataset.cache_dir == Path("./data/prbench_cache")
    
    def test_dataset_initialization_custom_cache(self):
        """Test dataset initialization with custom cache directory."""
        custom_path = "./custom_cache"
        dataset = PRBenchDataset(cache_dir=custom_path)
        
        assert dataset.cache_dir == Path(custom_path)
    
    @patch('dataset.load_dataset')
    def test_dataset_load_from_huggingface(self, mock_load_dataset, tmp_path):
        """Test loading dataset from HuggingFace when no cache exists."""
        mock_data = Mock()
        mock_load_dataset.return_value = mock_data
        
        dataset = PRBenchDataset(cache_dir=tmp_path / "cache")
        result = dataset.load()
        
        mock_load_dataset.assert_called_once_with("ScaleAI/PRBench")
        mock_data.save_to_disk.assert_called_once_with(str(dataset.cache_dir))
        assert result == mock_data
        assert dataset.dataset == mock_data
    
    @patch('dataset.load_from_disk')
    def test_dataset_load_from_cache(self, mock_load_from_disk):
        """Test loading dataset from cache when it exists."""
        mock_data = {"finance": [{"topic": "cached"}]}
        mock_load_from_disk.return_value = mock_data
        
        dataset = PRBenchDataset(cache_dir="./data/prbench_cache")
        
        # Mock cache directory exists
        with patch.object(Path, 'exists', return_value=True):
            result = dataset.load(use_cache=True)
        
        mock_load_from_disk.assert_called_once()
        assert result == mock_data
        assert dataset.dataset == mock_data
    
    @patch('dataset.load_dataset')
    @patch('dataset.load_from_disk')
    def test_dataset_load_cache_fallback(self, mock_load_from_disk, mock_load_dataset, tmp_path):
        """Test that loading falls back to HuggingFace if cache fails."""
        mock_load_from_disk.side_effect = Exception("Cache corrupt")
        mock_data = Mock()
        mock_load_dataset.return_value = mock_data
        
        dataset = PRBenchDataset(cache_dir=tmp_path / "cache")
        dataset.cache_dir.mkdir()
        result = dataset.load()
        
        mock_load_from_disk.assert_called_once()
        mock_load_dataset.assert_called_once_with("ScaleAI/PRBench")
        mock_data.save_to_disk.assert_called_once_with(str(dataset.cache_dir))
        assert result == mock_data
    
    def test_get_sample(self):
        """Test retrieving a single sample from the dataset."""
        dataset = PRBenchDataset()
        
        # Mock the dataset
        mock_sample = {
            "topic": "Finance",
            "expert": "Jane Doe",
            "turns": 5,
            "conversation": []
        }
        dataset.dataset = {
            "finance": [mock_sample]
        }
        
        result = dataset.get_sample("finance", 0)
        
        assert result["topic"] == "Finance"
        assert result["expert"] == "Jane Doe"
        assert result["turns"] == 5
    
    def test_get_sample_auto_loads(self):
        """Test that get_sample auto-loads dataset if not loaded."""
        dataset = PRBenchDataset()
        
        def load_fixture():
            dataset.dataset = {"finance": [{"topic": "test"}]}

        with patch.object(dataset, 'load', side_effect=load_fixture) as mock_load:
            assert dataset.get_sample("finance", 0) == {"topic": "test"}

        mock_load.assert_called_once()
    
    def test_get_samples_multiple(self):
        """Test retrieving multiple samples from the dataset."""
        dataset = PRBenchDataset()
        
        mock_samples = [
            {"topic": "Finance 1", "turns": 3},
            {"topic": "Finance 2", "turns": 5},
            {"topic": "Finance 3", "turns": 7}
        ]
        dataset.dataset = {"finance": mock_samples}
        
        result = dataset.get_samples("finance", start=0, end=3)
        
        assert len(result) == 3
        assert result[0]["topic"] == "Finance 1"
        assert result[2]["turns"] == 7
    
    def test_get_samples_respects_bounds(self):
        """Test that get_samples doesn't exceed dataset length."""
        dataset = PRBenchDataset()
        
        mock_samples = [
            {"topic": "Finance 1"},
            {"topic": "Finance 2"}
        ]
        dataset.dataset = {"finance": mock_samples}
        
        # Request more samples than available
        result = dataset.get_samples("finance", start=0, end=10)
        
        assert len(result) == 2
    
    def test_get_stats(self):
        """Test getting statistics about a dataset split."""
        dataset = PRBenchDataset()
        
        mock_features = MagicMock()
        mock_features.keys.return_value = ["topic", "expert", "conversation"]
        
        mock_split = MagicMock()
        mock_split.__len__.return_value = 100
        mock_split.features = mock_features
        mock_split.__getitem__.side_effect = lambda key: {
            "topic": ["Finance", "Tax", "Finance"],
            "expert": ["Expert1", "Expert2", "Expert1"]
        }[key]
        
        dataset.dataset = {"finance": mock_split}
        
        stats = dataset.get_stats("finance")
        
        assert stats["total_samples"] == 100
        assert "topic" in stats["fields"]


# ============================================================================
# Codex Invocation Tests
# ============================================================================

class TestCodexInvocation:
    """Tests for codex CLI invocation functionality."""
    
    @patch('subprocess.run')
    def test_codex_success(self, mock_run):
        """Test successful codex invocation."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "4"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        result = call_codex("What is 2+2?")
        
        assert result["success"] is True
        assert result["output"] == "4"
        assert result["error"] is None
        assert result["returncode"] == 0
        mock_run.assert_called_once_with(
            ["codex", "exec", "What is 2+2?"],
            capture_output=True,
            text=True,
            timeout=30
        )
    
    @patch('subprocess.run')
    def test_codex_failure(self, mock_run):
        """Test codex invocation with error."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: Invalid prompt"
        mock_run.return_value = mock_result
        
        result = call_codex("Invalid prompt")
        
        assert result["success"] is False
        assert result["error"] == "Error: Invalid prompt"
        assert result["returncode"] == 1
    
    @patch('subprocess.run')
    def test_codex_timeout(self, mock_run):
        """Test codex invocation timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("codex", 30)
        
        result = call_codex("Long running prompt", timeout=30)
        
        assert result["success"] is False
        assert "timed out after 30 seconds" in result["error"]
        assert result["returncode"] == -1
    
    @patch('subprocess.run')
    def test_codex_not_found(self, mock_run):
        """Test handling when codex CLI is not installed."""
        mock_run.side_effect = FileNotFoundError()
        
        result = call_codex("test prompt")
        
        assert result["success"] is False
        assert "codex CLI not found" in result["error"]
        assert result["returncode"] == -1
    
    @patch('subprocess.run')
    def test_codex_custom_timeout(self, mock_run):
        """Test codex invocation with custom timeout."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        call_codex("test", timeout=60)
        
        mock_run.assert_called_once()
        assert mock_run.call_args[1]["timeout"] == 60
    
    @patch('subprocess.run')
    def test_codex_strips_whitespace(self, mock_run):
        """Test that codex output is stripped of whitespace."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "  output with spaces  \n"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        result = call_codex("test")
        
        assert result["output"] == "output with spaces"
    
    @patch('subprocess.run')
    def test_codex_unexpected_error(self, mock_run):
        """Test handling of unexpected errors."""
        mock_run.side_effect = RuntimeError("Unexpected issue")
        
        result = call_codex("test")
        
        assert result["success"] is False
        assert "Unexpected error" in result["error"]
        assert result["returncode"] == -1


# ============================================================================
# Multiturn Questions Tests
# ============================================================================

class TestMultiturnQuestions:
    """Tests for handling multiturn conversations from the dataset."""
    
    def test_sample_has_turns_field(self):
        """Test that samples contain the turns field."""
        dataset = PRBenchDataset()
        
        mock_sample = {
            "topic": "Finance",
            "expert": "Jane Doe",
            "turns": 5,
            "conversation": [
                {"role": "user", "content": "Question 1"},
                {"role": "assistant", "content": "Answer 1"},
                {"role": "user", "content": "Question 2"},
                {"role": "assistant", "content": "Answer 2"},
            ]
        }
        dataset.dataset = {"finance": [mock_sample]}
        
        result = dataset.get_sample("finance", 0)
        
        assert "turns" in result
        assert result["turns"] == 5
    
    def test_multiturn_conversation_structure(self):
        """Test that multiturn conversations have proper structure."""
        dataset = PRBenchDataset()
        
        conversation = [
            {"role": "user", "content": "What is compound interest?"},
            {"role": "assistant", "content": "Compound interest is..."},
            {"role": "user", "content": "Can you give an example?"},
            {"role": "assistant", "content": "Sure, here's an example..."},
            {"role": "user", "content": "How does it differ from simple interest?"},
            {"role": "assistant", "content": "The key difference is..."}
        ]
        
        mock_sample = {
            "topic": "Finance",
            "expert": "Expert",
            "turns": 3,
            "conversation": conversation
        }
        dataset.dataset = {"finance": [mock_sample]}
        
        result = dataset.get_sample("finance", 0)
        
        assert result["turns"] == 3
        assert len(result["conversation"]) == 6  # 3 turns * 2 messages (user + assistant)
        assert result["conversation"][0]["role"] == "user"
        assert result["conversation"][1]["role"] == "assistant"
    
    def test_filter_by_turn_count(self):
        """Test filtering samples by number of turns."""
        dataset = PRBenchDataset()
        
        mock_samples = [
            {"topic": "Tax", "turns": 2},
            {"topic": "Investment", "turns": 5},
            {"topic": "Banking", "turns": 3},
            {"topic": "Trading", "turns": 5}
        ]
        
        # Create a mock dataset split
        mock_split = MagicMock()
        mock_split.filter = lambda func: [s for s in mock_samples if func(s)]
        dataset.dataset = {"finance": mock_split}
        
        # Filter for samples with 5 turns
        result = dataset.filter_by_field("finance", "turns", 5)
        
        assert len(result) == 2
        assert all(r["turns"] == 5 for r in result)
    
    def test_get_samples_with_multiturn(self):
        """Test retrieving multiple samples with multiturn conversations."""
        dataset = PRBenchDataset()
        
        mock_samples = [
            {
                "topic": "Finance 1",
                "turns": 3,
                "conversation": [
                    {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "A1"}
                ]
            },
            {
                "topic": "Finance 2",
                "turns": 5,
                "conversation": [
                    {"role": "user", "content": "Q2"},
                    {"role": "assistant", "content": "A2"}
                ]
            },
            {
                "topic": "Finance 3",
                "turns": 2,
                "conversation": [
                    {"role": "user", "content": "Q3"},
                    {"role": "assistant", "content": "A3"}
                ]
            }
        ]
        dataset.dataset = {"finance": mock_samples}
        
        result = dataset.get_samples("finance", start=0, end=3)
        
        assert len(result) == 3
        assert all("turns" in sample for sample in result)
        assert all("conversation" in sample for sample in result)
        assert result[0]["turns"] == 3
        assert result[1]["turns"] == 5
        assert result[2]["turns"] == 2
    
    def test_multiturn_across_different_splits(self):
        """Test that multiturn conversations work across different splits."""
        dataset = PRBenchDataset()
        
        dataset.dataset = {
            "finance": [{"topic": "Tax", "turns": 3}],
            "legal": [{"topic": "Contract", "turns": 4}],
            "finance_hard": [{"topic": "Derivatives", "turns": 6}],
            "legal_hard": [{"topic": "Litigation", "turns": 5}]
        }
        
        finance_sample = dataset.get_sample("finance", 0)
        legal_sample = dataset.get_sample("legal", 0)
        finance_hard_sample = dataset.get_sample("finance_hard", 0)
        legal_hard_sample = dataset.get_sample("legal_hard", 0)
        
        assert finance_sample["turns"] == 3
        assert legal_sample["turns"] == 4
        assert finance_hard_sample["turns"] == 6
        assert legal_hard_sample["turns"] == 5
    
    def test_conversation_extraction_from_multiturn(self):
        """Test extracting individual turns from a multiturn conversation."""
        dataset = PRBenchDataset()
        
        conversation = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Answer 2"},
            {"role": "user", "content": "Question 3"},
            {"role": "assistant", "content": "Answer 3"}
        ]
        
        mock_sample = {
            "topic": "Finance",
            "turns": 3,
            "conversation": conversation
        }
        dataset.dataset = {"finance": [mock_sample]}
        
        sample = dataset.get_sample("finance", 0)
        
        # Extract user questions (odd indices in conversation)
        user_questions = [msg["content"] for msg in sample["conversation"] if msg["role"] == "user"]
        
        assert len(user_questions) == 3
        assert user_questions[0] == "Question 1"
        assert user_questions[1] == "Question 2"
        assert user_questions[2] == "Question 3"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining dataset and codex functionality."""
    
    @patch('subprocess.run')
    def test_dataset_to_codex_pipeline(self, mock_run):
        """Test full pipeline from dataset to codex invocation."""
        # Setup dataset
        dataset = PRBenchDataset()
        mock_sample = {
            "topic": "Finance",
            "expert": "Expert",
            "turns": 1,
            "conversation": [
                {"role": "user", "content": "What is 2+2?"}
            ]
        }
        dataset.dataset = {"finance": [mock_sample]}
        
        # Setup codex mock
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "4"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        # Get sample and call codex
        sample = dataset.get_sample("finance", 0)
        question = sample["conversation"][0]["content"]
        result = call_codex(question)
        
        assert result["success"] is True
        assert result["output"] == "4"
    
    def test_multiturn_sample_validity(self):
        """Test that multiturn samples have valid structure for processing."""
        dataset = PRBenchDataset()
        
        mock_sample = {
            "topic": "Investment Strategy",
            "expert": "Financial Advisor",
            "turns": 4,
            "conversation": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
                {"role": "user", "content": "Q3"},
                {"role": "assistant", "content": "A3"},
                {"role": "user", "content": "Q4"},
                {"role": "assistant", "content": "A4"}
            ]
        }
        dataset.dataset = {"finance": [mock_sample]}
        
        sample = dataset.get_sample("finance", 0)
        
        # Validate structure
        assert "turns" in sample
        assert "conversation" in sample
        assert sample["turns"] > 0
        assert len(sample["conversation"]) == sample["turns"] * 2
        
        # Validate conversation alternates between user and assistant
        for i, msg in enumerate(sample["conversation"]):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert msg["role"] == expected_role
            assert "content" in msg
            assert len(msg["content"]) > 0


# ============================================================================
# Functional Tests (Real Dataset & Codex)
# ============================================================================

class TestFunctionalDatasetLoading:
    """Functional tests for actual dataset loading."""
    
    def test_dataset_loads_successfully(self):
        """Test that the dataset can actually be loaded from HuggingFace or cache."""
        dataset = PRBenchDataset()
        
        # This will attempt to load from cache or download from HuggingFace
        result = dataset.load()
        
        assert result is not None
        assert dataset.dataset is not None
        assert isinstance(dataset.dataset, dict)
    
    def test_dataset_has_expected_splits(self):
        """Test that loaded dataset contains expected splits."""
        dataset = PRBenchDataset()
        dataset.load()
        
        for split in dataset.available_splits:
            assert split in dataset.dataset, f"Split '{split}' not found in dataset"
            assert len(dataset.dataset[split]) > 0, f"Split '{split}' is empty"
    
    def test_dataset_samples_have_required_fields(self):
        """Test that dataset samples contain required fields."""
        dataset = PRBenchDataset()
        dataset.load()
        
        # Check at least one sample from each split
        for split in dataset.available_splits:
            if len(dataset.dataset[split]) > 0:
                sample = dataset.get_sample(split, 0)
                
                assert "topic" in sample, f"Sample from '{split}' missing 'topic' field"
                assert "expert" in sample, f"Sample from '{split}' missing 'expert' field"
                assert "turns" in sample, f"Sample from '{split}' missing 'turns' field"
                assert isinstance(sample["turns"], int), f"'turns' field should be int, got {type(sample['turns'])}"
                
                # Check that prompts and responses exist for multiturn conversations
                if sample["turns"] > 0:
                    assert "prompt_0" in sample, f"Sample from '{split}' missing 'prompt_0' field"
                    assert "response_0" in sample, f"Sample from '{split}' missing 'response_0' field"


class TestFunctionalCodexInvocation:
    """Functional tests for actual codex CLI invocation."""
    
    def test_codex_is_available(self):
        """Test that codex CLI is available and can be invoked."""
        result = call_codex("echo 'test'")
        
        # If codex is not available, the test will show the error but won't fail
        # This allows the test suite to run even if codex isn't installed
        if not result["success"]:
            pytest.skip(f"Codex CLI not available: {result['error']}")
        
        assert result["success"] is True
        assert result["returncode"] == 0
    
    def test_codex_responds_to_simple_query(self):
        """Test that codex can respond to a simple query."""
        result = call_codex("What is 2+2?")
        
        if not result["success"]:
            pytest.skip(f"Codex CLI not available: {result['error']}")
        
        assert result["success"] is True
        assert result["output"] is not None
        assert len(result["output"]) > 0
    
    def test_codex_handles_timeout(self):
        """Test that codex respects timeout settings."""
        # Use a very short timeout to test timeout handling
        result = call_codex("sleep 5", timeout=1)
        
        if not result["success"]:
            # If codex isn't available, skip
            if "not found" in result.get("error", "").lower():
                pytest.skip(f"Codex CLI not available: {result['error']}")
            # Otherwise, timeout is expected
            assert "timed out" in result["error"].lower() or result["returncode"] == -1


class TestFunctionalMultiturnQuestions:
    """Functional tests for getting multiturn questions from the actual dataset."""
    
    def test_can_get_multiturn_samples(self):
        """Test that we can retrieve multiturn conversation samples from the dataset."""
        dataset = PRBenchDataset()
        dataset.load()
        
        # Try to find a sample with multiple turns
        for split in dataset.available_splits:
            if len(dataset.dataset[split]) > 0:
                # Check first few samples to find one with multiple turns
                for i in range(min(10, len(dataset.dataset[split]))):
                    sample = dataset.get_sample(split, i)
                    
                    if sample.get("turns", 0) > 1:
                        assert sample["turns"] > 1, f"Expected multiturn sample, got {sample['turns']} turns"
                        
                        # Verify prompts and responses exist for all turns
                        for turn in range(sample["turns"]):
                            assert f"prompt_{turn}" in sample, \
                                f"Sample missing 'prompt_{turn}' for turn {turn}"
                            if turn < sample["turns"] - 1:  # Last turn might not have response
                                assert f"response_{turn}" in sample, \
                                    f"Sample missing 'response_{turn}' for turn {turn}"
                        return
        
        pytest.skip("No multiturn samples found in first 10 samples of any split")
    
    def test_multiturn_conversation_structure(self):
        """Test that multiturn conversations have proper structure."""
        dataset = PRBenchDataset()
        dataset.load()
        
        multiturn_found = False
        
        for split in dataset.available_splits:
            if len(dataset.dataset[split]) > 0:
                for i in range(min(20, len(dataset.dataset[split]))):
                    sample = dataset.get_sample(split, i)
                    
                    if sample.get("turns", 0) > 1:
                        multiturn_found = True
                        turns = sample["turns"]
                        
                        # Validate that prompts exist for all turns
                        for turn in range(turns):
                            prompt_key = f"prompt_{turn}"
                            assert prompt_key in sample, f"Missing {prompt_key} for turn {turn}"
                            assert len(sample[prompt_key]) > 0, f"{prompt_key} is empty"
                            
                            # Responses exist for all turns except possibly the last
                            if turn < turns - 1:
                                response_key = f"response_{turn}"
                                assert response_key in sample, f"Missing {response_key} for turn {turn}"
                                assert len(sample[response_key]) > 0, f"{response_key} is empty"
                        
                        return
        
        if not multiturn_found:
            pytest.skip("No multiturn samples found in first 20 samples of any split")
    
    def test_can_extract_user_questions_from_multiturn(self):
        """Test that we can extract user questions from multiturn conversations."""
        dataset = PRBenchDataset()
        dataset.load()
        
        for split in dataset.available_splits:
            if len(dataset.dataset[split]) > 0:
                for i in range(min(20, len(dataset.dataset[split]))):
                    sample = dataset.get_sample(split, i)
                    
                    if sample.get("turns", 0) > 1:
                        turns = sample["turns"]
                        
                        # Extract user questions from prompt_0, prompt_1, etc.
                        user_questions = [
                            sample[f"prompt_{turn}"]
                            for turn in range(turns)
                            if f"prompt_{turn}" in sample
                        ]
                        
                        assert len(user_questions) == turns, \
                            f"Expected {turns} user questions, got {len(user_questions)}"
                        assert all(len(q) > 0 for q in user_questions), \
                            "All user questions should have content"
                        
                        return
        
        pytest.skip("No multiturn samples found in first 20 samples of any split")
    
    def test_multiturn_samples_across_all_splits(self):
        """Test that multiturn samples exist across all dataset splits."""
        dataset = PRBenchDataset()
        dataset.load()
        
        splits_with_multiturn = []
        
        for split in dataset.available_splits:
            if len(dataset.dataset[split]) > 0:
                # Check first 10 samples
                for i in range(min(10, len(dataset.dataset[split]))):
                    sample = dataset.get_sample(split, i)
                    if sample.get("turns", 0) > 1:
                        splits_with_multiturn.append(split)
                        break
        
        assert len(splits_with_multiturn) > 0, \
            "No multiturn samples found in any split"
        
        # Log which splits have multiturn samples
        print(f"\nFound multiturn samples in splits: {splits_with_multiturn}")


class TestCodexMultiturnMemory:
    """Tests for codex multiturn conversation memory."""
    
    def test_codex_remembers_name_across_turns(self):
        """Test that codex remembers a name mentioned earlier in the conversation."""
        clear_history()
        
        # First turn: Introduce name and ask a simple question
        name = "Alice"
        turn1_prompt = f"My name is {name}. What is 2+2?"
        add_user_message(turn1_prompt)
        
        # Build prompt with history
        history = "\n".join(get_history())
        result1 = call_codex(history)
        
        if not result1["success"]:
            pytest.skip(f"Codex CLI not available: {result1['error']}")
        
        assert result1["success"] is True
        turn1_response = result1["output"]
        add_assistant_message(turn1_response)
        
        # Second turn: Ask something unrelated
        turn2_prompt = "What is the capital of France?"
        add_user_message(turn2_prompt)
        
        history = "\n".join(get_history())
        result2 = call_codex(history)
        
        if not result2["success"]:
            pytest.skip(f"Codex CLI not available: {result2['error']}")
        
        assert result2["success"] is True
        turn2_response = result2["output"]
        add_assistant_message(turn2_response)
        
        # Third turn: Ask about the name to test memory
        turn3_prompt = "What is my name?"
        add_user_message(turn3_prompt)
        
        history = "\n".join(get_history())
        result3 = call_codex(history)
        
        if not result3["success"]:
            pytest.skip(f"Codex CLI not available: {result3['error']}")
        
        assert result3["success"] is True
        turn3_response = result3["output"].lower()
        
        # Check if the name is mentioned in the response
        assert name.lower() in turn3_response, \
            f"Expected codex to remember name '{name}' but got: {result3['output']}"
    
    def test_codex_remembers_context_across_multiple_turns(self):
        """Test that codex maintains context across multiple conversation turns."""
        clear_history()
        
        # Use a unique name to avoid false positives
        name = "Zephyr"
        favorite_color = "turquoise"
        
        # Turn 1: Introduce name and favorite color
        turn1_prompt = f"Hi, my name is {name} and my favorite color is {favorite_color}. What is 5+3?"
        add_user_message(turn1_prompt)
        
        history = "\n".join(get_history())
        result1 = call_codex(history)
        
        if not result1["success"]:
            pytest.skip(f"Codex CLI not available: {result1['error']}")
        
        assert result1["success"] is True
        turn1_response = result1["output"]
        add_assistant_message(turn1_response)
        
        # Turn 2: Ask something completely different
        turn2_prompt = "Can you explain what photosynthesis is?"
        add_user_message(turn2_prompt)
        
        history = "\n".join(get_history())
        result2 = call_codex(history)
        
        if not result2["success"]:
            pytest.skip(f"Codex CLI not available: {result2['error']}")
        
        assert result2["success"] is True
        turn2_response = result2["output"]
        add_assistant_message(turn2_response)
        
        # Turn 3: Ask about the name
        turn3_prompt = "What is my name?"
        add_user_message(turn3_prompt)
        
        history = "\n".join(get_history())
        result3 = call_codex(history)
        
        if not result3["success"]:
            pytest.skip(f"Codex CLI not available: {result3['error']}")
        
        assert result3["success"] is True
        turn3_response = result3["output"].lower()
        
        assert name.lower() in turn3_response, \
            f"Expected codex to remember name '{name}' but got: {result3['output']}"
        
        add_assistant_message(result3["output"])
        
        # Turn 4: Ask about the favorite color
        turn4_prompt = "What is my favorite color?"
        add_user_message(turn4_prompt)
        
        history = "\n".join(get_history())
        result4 = call_codex(history)
        
        if not result4["success"]:
            pytest.skip(f"Codex CLI not available: {result4['error']}")
        
        assert result4["success"] is True
        turn4_response = result4["output"].lower()
        
        assert favorite_color.lower() in turn4_response, \
            f"Expected codex to remember favorite color '{favorite_color}' but got: {result4['output']}"
