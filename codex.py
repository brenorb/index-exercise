"""Codex CLI interface module."""

import subprocess
from typing import Dict, List, Optional

# Ephemeral chat memory
_chat_history: List[str] = []


def add_user_message(message: str) -> None:
    """Add a user message to the chat history."""
    _chat_history.append(f"User: {message}")


def add_assistant_message(message: str) -> None:
    """Add an assistant message to the chat history."""
    _chat_history.append(f"Assistant: {message}")


def clear_history() -> None:
    """Erase all chat history."""
    _chat_history.clear()


def get_history() -> List[str]:
    """Get the current chat history."""
    return _chat_history.copy()


def call_codex(prompt: str, timeout: int = 30) -> Dict[str, any]:
    """
    Calls the codex CLI with the given prompt and parses the output.
    
    Args:
        prompt: The prompt to send to codex
        timeout: Command timeout in seconds (default: 30)
        
    Returns:
        dict with 'success', 'output', 'error', and 'returncode' keys
    """
    try:
        result = subprocess.run(
            ["codex", "exec", prompt],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        return {
            "success": result.returncode == 0,
            "output": result.stdout.strip(),
            "error": result.stderr.strip() if result.stderr else None,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": None,
            "error": f"Command timed out after {timeout} seconds",
            "returncode": -1
        }
    except FileNotFoundError:
        return {
            "success": False,
            "output": None,
            "error": "codex CLI not found. Make sure it's installed and in your PATH.",
            "returncode": -1
        }
    except Exception as e:
        return {
            "success": False,
            "output": None,
            "error": f"Unexpected error: {str(e)}",
            "returncode": -1
        }

if __name__ == "__main__":
    result = call_codex("Hello, what is 2+2? and btw, my name is alicia")
    print(result)
    result2 = call_codex("What is my name?")
    print(result2)