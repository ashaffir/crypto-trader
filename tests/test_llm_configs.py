"""Tests for LLM configuration management."""

import json
import os
import tempfile

import pytest

from ui.lib.settings_state import (
    load_llm_configs,
    save_llm_configs,
    upsert_llm_config,
    delete_llm_config,
    set_active_llm,
    get_active_llm_config,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_load_empty_llm_configs(temp_dir):
    """Test loading LLM configs when file doesn't exist."""
    configs = load_llm_configs(temp_dir)
    assert configs == []


def test_save_and_load_llm_configs(temp_dir):
    """Test saving and loading LLM configurations."""
    test_configs = [
        {
            "name": "openai-gpt4",
            "provider": "OpenAI",
            "model": "gpt-4",
            "api_key": "test-key",
            "base_url": "https://api.openai.com",
            "system_prompt": "You are a trading assistant",
            "user_template": "Analyze: {{DATA_WINDOW}}",
            "is_active": True,
        },
        {
            "name": "claude",
            "provider": "Anthropic",
            "model": "claude-3-opus",
            "api_key": "test-key-2",
            "base_url": "https://api.anthropic.com",
            "system_prompt": "Trading expert",
            "user_template": "Review: {{DATA_WINDOW}}",
            "is_active": False,
        },
    ]

    assert save_llm_configs(test_configs, temp_dir)
    loaded = load_llm_configs(temp_dir)

    assert len(loaded) == 2
    assert loaded[0]["name"] == "openai-gpt4"
    assert loaded[1]["name"] == "claude"
    assert loaded[0]["is_active"] is True
    assert loaded[1]["is_active"] is False


def test_upsert_new_llm_config(temp_dir):
    """Test creating a new LLM config."""
    new_config = {
        "name": "test-llm",
        "provider": "Ollama",
        "model": "test-model",
        "api_key": "key123",
        "base_url": "https://llm.actappon.com",
        "system_prompt": "Test system",
        "user_template": "Test user",
        "is_active": False,
    }

    assert upsert_llm_config(new_config, temp_dir)

    configs = load_llm_configs(temp_dir)
    assert len(configs) == 1
    assert configs[0]["name"] == "test-llm"
    assert configs[0]["provider"] == "Ollama"


def test_upsert_update_existing_llm_config(temp_dir):
    """Test updating an existing LLM config."""
    # Create initial config
    initial = {
        "name": "test-llm",
        "provider": "Provider1",
        "model": "model1",
        "api_key": "key1",
        "base_url": "https://test1.com",
        "system_prompt": "Prompt1",
        "user_template": "Template1",
        "is_active": False,
    }
    assert upsert_llm_config(initial, temp_dir)

    # Update the config
    updated = {
        "name": "test-llm",
        "provider": "Provider2",
        "model": "model2",
        "api_key": "key2",
        "base_url": "https://test2.com",
        "system_prompt": "Prompt2",
        "user_template": "Template2",
        "is_active": True,
    }
    assert upsert_llm_config(updated, temp_dir)

    # Verify only one config exists with updated values
    configs = load_llm_configs(temp_dir)
    assert len(configs) == 1
    assert configs[0]["provider"] == "Provider2"
    assert configs[0]["is_active"] is True


def test_delete_llm_config(temp_dir):
    """Test deleting an LLM config."""
    # Create two configs
    configs = [
        {"name": "llm1", "provider": "P1", "model": "m1", "is_active": False},
        {"name": "llm2", "provider": "P2", "model": "m2", "is_active": True},
    ]
    assert save_llm_configs(configs, temp_dir)

    # Delete llm1
    assert delete_llm_config("llm1", temp_dir)

    remaining = load_llm_configs(temp_dir)
    assert len(remaining) == 1
    assert remaining[0]["name"] == "llm2"


def test_delete_nonexistent_config(temp_dir):
    """Test deleting a config that doesn't exist."""
    configs = [{"name": "llm1", "provider": "P1", "model": "m1"}]
    assert save_llm_configs(configs, temp_dir)

    # Try to delete non-existent config
    result = delete_llm_config("nonexistent", temp_dir)
    assert not result

    # Original config should still be there
    remaining = load_llm_configs(temp_dir)
    assert len(remaining) == 1


def test_set_active_llm(temp_dir):
    """Test setting an LLM as active."""
    # Create three configs
    configs = [
        {"name": "llm1", "provider": "P1", "model": "m1", "is_active": True},
        {"name": "llm2", "provider": "P2", "model": "m2", "is_active": False},
        {"name": "llm3", "provider": "P3", "model": "m3", "is_active": False},
    ]
    assert save_llm_configs(configs, temp_dir)

    # Set llm2 as active
    assert set_active_llm("llm2", temp_dir)

    updated = load_llm_configs(temp_dir)
    assert updated[0]["is_active"] is False  # llm1
    assert updated[1]["is_active"] is True  # llm2
    assert updated[2]["is_active"] is False  # llm3


def test_set_active_nonexistent_llm(temp_dir):
    """Test setting a non-existent LLM as active."""
    configs = [{"name": "llm1", "provider": "P1", "model": "m1", "is_active": True}]
    assert save_llm_configs(configs, temp_dir)

    # Try to set non-existent LLM as active
    result = set_active_llm("nonexistent", temp_dir)
    assert not result


def test_get_active_llm_config(temp_dir):
    """Test getting the active LLM config."""
    configs = [
        {"name": "llm1", "provider": "P1", "model": "m1", "is_active": False},
        {"name": "llm2", "provider": "P2", "model": "m2", "is_active": True},
        {"name": "llm3", "provider": "P3", "model": "m3", "is_active": False},
    ]
    assert save_llm_configs(configs, temp_dir)

    active = get_active_llm_config(temp_dir)
    assert active is not None
    assert active["name"] == "llm2"


def test_get_active_llm_config_no_active(temp_dir):
    """Test getting active LLM when none is marked active."""
    configs = [
        {"name": "llm1", "provider": "P1", "model": "m1", "is_active": False},
        {"name": "llm2", "provider": "P2", "model": "m2", "is_active": False},
    ]
    assert save_llm_configs(configs, temp_dir)

    # Should return first config as fallback
    active = get_active_llm_config(temp_dir)
    assert active is not None
    assert active["name"] == "llm1"


def test_get_active_llm_config_empty(temp_dir):
    """Test getting active LLM when no configs exist."""
    active = get_active_llm_config(temp_dir)
    assert active is None


def test_upsert_without_name(temp_dir):
    """Test upsert fails without a name."""
    config = {
        "provider": "TestProvider",
        "model": "test-model",
    }

    result = upsert_llm_config(config, temp_dir)
    assert not result

    configs = load_llm_configs(temp_dir)
    assert len(configs) == 0


def test_llm_config_file_location(temp_dir):
    """Test that configs are saved to the correct file."""
    config = {
        "name": "test",
        "provider": "P",
        "model": "m",
        "is_active": False,
    }
    upsert_llm_config(config, temp_dir)

    # Verify file exists
    config_file = os.path.join(temp_dir, "llm_configs.json")
    assert os.path.exists(config_file)

    # Verify structure
    with open(config_file, "r") as f:
        data = json.load(f)

    assert "configs" in data
    assert isinstance(data["configs"], list)
    assert len(data["configs"]) == 1
