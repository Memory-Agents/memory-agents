"""Tests for ChromaDB collection clearing functionality.

This module contains tests to verify that the ChromaDB collection can be
properly cleared and reset. The test ensures that the clear_collection
method successfully removes all stored conversations while maintaining
the ability to add new conversations afterward.

The test verifies:
- Initial empty state of the collection
- Proper addition of conversation turns
- Complete clearing of the collection
- Continued functionality after clearing
"""

import pytest
import os
from memory_agents.core.agents.baseline_vdb import BaselineVDBAgent
import shutil


@pytest.fixture
def temp_chroma_dir(tmp_path):
    """Provides a temporary directory for ChromaDB persistence.

    Creates a temporary directory for ChromaDB storage during testing
    and ensures it's cleaned up after the test completes.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.

    Yields:
        str: Path to the temporary ChromaDB directory.

    Side Effects:
        Creates and removes temporary directory for ChromaDB storage.
    """
    yield str(tmp_path / "test_chromadb")
    # Clean up the directory after the test
    if os.path.exists(str(tmp_path / "test_chromadb")):
        shutil.rmtree(str(tmp_path / "test_chromadb"))


def test_clear_collection_resets_conversations(temp_chroma_dir):
    """Test ChromaDB collection clearing and reset functionality.

    Tests that the clear_collection method successfully removes all
    conversations from the ChromaDB collection and that the collection
    remains functional for adding new conversations afterward.

    The test follows this sequence:
    1. Verify initial empty state
    2. Add multiple conversation turns
    3. Verify conversations were added
    4. Clear the collection
    5. Verify collection is empty
    6. Add new conversation to verify continued functionality

    Args:
        temp_chroma_dir: Path to temporary ChromaDB directory from fixture.

    Returns:
        None: Raises AssertionError if collection clearing fails.

    Raises:
        AssertionError: If conversation counts don't match expected values
            at any stage of the test.
    """
    # Initialize BaselineAgent with a temporary persistence directory
    agent = BaselineVDBAgent(persist_directory=temp_chroma_dir)
    chroma_manager = agent.chroma_manager

    # Ensure the collection is initially empty
    initial_count = chroma_manager.conversation_collection.count()
    assert initial_count == 0, (
        f"Expected initial count to be 0, but got {initial_count}"
    )

    # Add some dummy conversation turns
    chroma_manager.add_conversation_turn(
        user_message="Hello, how are you?",
        ai_message="I'm doing great, thanks for asking!",
        metadata={"thread_id": "test_thread_1"},
    )
    chroma_manager.add_conversation_turn(
        user_message="What's the weather like?",
        ai_message="I'm sorry, I don't have access to real-time weather information.",
        metadata={"thread_id": "test_thread_1"},
    )
    chroma_manager.add_conversation_turn(
        user_message="Tell me a joke.",
        ai_message="Why don't scientists trust atoms? Because they make up everything!",
        metadata={"thread_id": "test_thread_2"},
    )

    # Verify that conversations were added
    count_after_addition = chroma_manager.conversation_collection.count()
    assert count_after_addition == 3, (
        f"Expected 3 conversations, but got {count_after_addition}"
    )

    # Clear the collection
    chroma_manager.clear_collection()

    # Verify that the collection is empty after clearing
    count_after_clear = chroma_manager.conversation_collection.count()
    assert count_after_clear == 0, (
        f"Expected 0 conversations after clear, but got {count_after_clear}"
    )

    # Try adding again after clearing to ensure functionality
    chroma_manager.add_conversation_turn(
        user_message="New conversation after clear.",
        ai_message="Indeed, a fresh start.",
        metadata={"thread_id": "test_thread_3"},
    )
    count_after_re_addition = chroma_manager.conversation_collection.count()
    assert count_after_re_addition == 1, (
        f"Expected 1 conversation after re-addition, but got {count_after_re_addition}"
    )
