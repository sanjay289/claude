#!/usr/bin/env python3
"""Hermes agent loop with tool execution via Ollama."""

import json
import subprocess
import datetime
import os
import ollama

MODEL = "hermes3"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command and return output",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Get current date and time",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List contents of a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list"}
                }
            }
        }
    }
]


def run_shell(command: str) -> str:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        output = result.stdout + result.stderr
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out"
    except Exception as e:
        return f"Error: {e}"


def read_file(path: str) -> str:
    try:
        with open(os.path.expanduser(path), "r") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"


def write_file(path: str, content: str) -> str:
    try:
        with open(os.path.expanduser(path), "w") as f:
            f.write(content)
        return f"Written to {path}"
    except Exception as e:
        return f"Error: {e}"


def get_datetime() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def list_dir(path: str = '.') -> str:
    path = path or '.'
    try:
        entries = os.listdir(os.path.expanduser(path))
        return "\n".join(entries)
    except Exception as e:
        return f"Error: {e}"


TOOL_MAP = {
    "run_shell": run_shell,
    "read_file": read_file,
    "write_file": write_file,
    "get_datetime": get_datetime,
    "list_dir": list_dir,
}


def call_tool(name: str, args: dict) -> str:
    fn = TOOL_MAP.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    print(f"\n  [tool] {name}({json.dumps(args)})")
    result = fn(**args)
    print(f"  [result] {result[:200]}{'...' if len(result) > 200 else ''}")
    return result


def agent_loop(user_input: str, max_iterations: int = 10) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant with access to tools. "
                "Use tools when needed to answer the user's request. "
                "Think step by step and call tools as required."
            )
        },
        {"role": "user", "content": user_input}
    ]

    for i in range(max_iterations):
        print(f"\n[iteration {i+1}]")
        response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
        msg = response.message

        # Add assistant response to history
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})

        if not msg.tool_calls:
            return msg.content or "(no response)"

        # Execute each tool call
        for tc in msg.tool_calls:
            name = tc.function.name
            args = tc.function.arguments if isinstance(tc.function.arguments, dict) else json.loads(tc.function.arguments)
            result = call_tool(name, args)
            messages.append({
                "role": "tool",
                "content": result,
                "name": name
            })

    return "Max iterations reached."


def main():
    print(f"Hermes Agent — model: {MODEL}")
    print("Type 'quit' to exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        print("\nAgent thinking...")
        answer = agent_loop(user_input)
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()
