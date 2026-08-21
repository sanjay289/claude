#!/usr/bin/env python3
"""Hermes agent loop with tool execution via Ollama."""

import json
import re
import subprocess
import datetime
import os
import ollama

MODEL = "hermes3"
MAX_FAKE_TOOL_CALL_RETRIES = 2

SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")


def parse_skill_file(path: str) -> dict:
    with open(path, "r") as f:
        text = f.read()
    meta = {"name": os.path.splitext(os.path.basename(path))[0], "description": ""}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            frontmatter = text[3:end].strip()
            body = text[end + 4:].lstrip("\n")
            for line in frontmatter.splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    meta[key.strip()] = value.strip()
    meta["body"] = body.strip()
    return meta


def discover_skills() -> dict:
    skills = {}
    if not os.path.isdir(SKILLS_DIR):
        return skills
    for fname in sorted(os.listdir(SKILLS_DIR)):
        if fname.endswith(".md"):
            skill = parse_skill_file(os.path.join(SKILLS_DIR, fname))
            skills[skill["name"]] = skill
    return skills


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
    },
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": "List available skills in the skill hub, with a one-line description of each",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "use_skill",
            "description": "Load a skill's full instructions by name so they can be followed for the current task",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Skill name, as returned by list_skills"}
                },
                "required": ["name"]
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


def list_skills() -> str:
    skills = discover_skills()
    if not skills:
        return "(no skills found)"
    return "\n".join(f"{name}: {s['description']}" for name, s in skills.items())


def use_skill(name: str) -> str:
    skills = discover_skills()
    skill = skills.get(name)
    if not skill:
        available = ", ".join(skills) or "(none)"
        return f"Unknown skill '{name}'. Available: {available}"
    return skill["body"]


TOOL_MAP = {
    "run_shell": run_shell,
    "read_file": read_file,
    "write_file": write_file,
    "get_datetime": get_datetime,
    "list_dir": list_dir,
    "list_skills": list_skills,
    "use_skill": use_skill,
}


def looks_like_fake_tool_call(content: str) -> bool:
    """True if content is prose containing a JSON blob that names a real tool,
    i.e. the model wrote a tool call as text instead of calling it for real."""
    if not content:
        return False
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return False
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    name = data.get("name") or data.get("tool") or data.get("function")
    return isinstance(name, str) and name in TOOL_MAP


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
                "You also have a skill hub: call list_skills to see what's "
                "available, and use_skill(name) to load a skill's instructions "
                "before doing a task it covers (e.g. summarizing, reviewing code, "
                "or running a risky shell command). Prefer loading a matching "
                "skill over guessing at the right approach. "
                "Think step by step and call tools as required."
            )
        },
        {"role": "user", "content": user_input}
    ]
    fake_tool_call_retries = 0

    for i in range(max_iterations):
        print(f"\n[iteration {i+1}]")
        response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
        msg = response.message

        # Add assistant response to history
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})

        if not msg.tool_calls:
            if looks_like_fake_tool_call(msg.content) and fake_tool_call_retries < MAX_FAKE_TOOL_CALL_RETRIES:
                fake_tool_call_retries += 1
                print(f"  [warn] tool call written as text, asking model to retry "
                      f"({fake_tool_call_retries}/{MAX_FAKE_TOOL_CALL_RETRIES})")
                messages.append({
                    "role": "user",
                    "content": (
                        "That was a tool call written as text instead of an actual tool "
                        "call. Call the tool for real using the tool-calling interface."
                    )
                })
                continue
            if looks_like_fake_tool_call(msg.content):
                print(f"  [warn] still getting fake tool calls after "
                      f"{MAX_FAKE_TOOL_CALL_RETRIES} retries, giving up and "
                      f"returning raw text")
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
