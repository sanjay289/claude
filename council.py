#!/usr/bin/env python3
"""Claude council: parallel multi-model deliberation via Ollama.

Each member model answers the prompt independently and in parallel, then a
separate chair model reads every answer and writes a synthesis noting where
the council agrees and disagrees.
"""

import concurrent.futures
import subprocess
import time

import ollama

MEMBERS = ["gpt-oss:120b-cloud", "minimax-m3:cloud", "llama3.2:3b"]
CHAIR = "nemotron-3-ultra:cloud"

COLORS = {
    "gpt-oss:120b-cloud": "\033[32m",
    "minimax-m3:cloud": "\033[34m",
    "llama3.2:3b": "\033[33m",
}
CHAIR_COLOR = "\033[1;37m"
RESET = "\033[0m"


def ensure_server_running(timeout: float = 15.0) -> None:
    try:
        ollama.list()
        return
    except Exception:
        pass
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            ollama.list()
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("ollama server did not come up in time")


def ask_member(model: str, question: str) -> str:
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": question}],
        )
        return response.message.content or "(empty response)"
    except Exception as e:
        return f"(error: {e})"


def gather_opinions(question: str) -> dict:
    opinions = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(MEMBERS)) as pool:
        futures = {pool.submit(ask_member, m, question): m for m in MEMBERS}
        for future in concurrent.futures.as_completed(futures):
            model = futures[future]
            opinions[model] = future.result()
            color = COLORS.get(model, "")
            print(f"{color}[{model}]{RESET}\n{opinions[model]}\n")
    return opinions


def synthesize(question: str, opinions: dict) -> str:
    transcript = "\n\n".join(
        f"--- {model} ---\n{answer}" for model, answer in opinions.items()
    )
    chair_prompt = (
        f"A user asked the following question:\n\n{question}\n\n"
        f"Council members answered independently:\n\n{transcript}\n\n"
        "As chair, synthesize a final answer. Note where the council agrees, "
        "call out any real disagreements and why they might differ, then give "
        "your best consensus recommendation. Be concise."
    )
    try:
        response = ollama.chat(
            model=CHAIR,
            messages=[{"role": "user", "content": chair_prompt}],
        )
        return response.message.content or "(empty response)"
    except Exception as e:
        return f"(error: {e})"


def run_council(question: str) -> None:
    print(f"\nConvening council on: {question}\n")
    opinions = gather_opinions(question)

    print(f"{CHAIR_COLOR}[chair: {CHAIR}] synthesizing...{RESET}\n")
    verdict = synthesize(question, opinions)
    print(f"{CHAIR_COLOR}[verdict]{RESET}\n{verdict}\n")


def main():
    print(f"Claude council — members: {', '.join(MEMBERS)} | chair: {CHAIR}")
    print("Type 'quit' to exit\n")

    ensure_server_running()

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        run_council(question)


if __name__ == "__main__":
    main()
