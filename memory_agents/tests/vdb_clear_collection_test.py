import pytest
import os
from memory_agents.core.agents.baseline_vdb import BaselineVDBAgent
from memory_agents.core.chroma_db_manager import ChromaDBManager
import shutil


@pytest.fixture
def temp_chroma_dir(tmp_path):
    """Provides a temporary directory for ChromaDB persistence."""
    yield str(tmp_path / "test_chromadb")
    # Clean up the directory after the test
    if os.path.exists(str(tmp_path / "test_chromadb")):
        shutil.rmtree(str(tmp_path / "test_chromadb"))


def test_clear_collection_resets_conversations(temp_chroma_dir):
    """
    Tests that clear_collection successfully prunes and resets all conversations
    within the ChromaDB collection.
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
        assistant_message="I'm doing great, thanks for asking!",
        metadata={"thread_id": "test_thread_1"},
    )
    chroma_manager.add_conversation_turn(
        user_message="What's the weather like?",
        assistant_message="I'm sorry, I don't have access to real-time weather information.",
        metadata={"thread_id": "test_thread_1"},
    )
    chroma_manager.add_conversation_turn(
        user_message="Tell me a joke.",
        assistant_message="Why don't scientists trust atoms? Because they make up everything!",
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
        assistant_message="Indeed, a fresh start.",
        metadata={"thread_id": "test_thread_3"},
    )
    count_after_re_addition = chroma_manager.conversation_collection.count()
    assert count_after_re_addition == 1, (
        f"Expected 1 conversation after re-addition, but got {count_after_re_addition}"
    )
