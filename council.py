#!/usr/bin/env python3
"""Claude council: parallel multi-model deliberation via Ollama.

Built on graph_engine.Graph: a fan-out node asks every member model in
parallel, then joins into a chair node that synthesizes a final answer.
"""

import subprocess
import time

import ollama

from graph_engine import Graph, END

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


def make_ask_node(model: str):
    def ask(state: dict) -> str:
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": state["question"]}],
            )
            answer = response.message.content or "(empty response)"
        except Exception as e:
            answer = f"(error: {e})"
        color = COLORS.get(model, "")
        print(f"{color}[{model}]{RESET}\n{answer}\n")
        return answer
    return ask


def synthesize_node(state: dict) -> dict:
    print(f"{CHAIR_COLOR}[chair: {CHAIR}] synthesizing...{RESET}\n")
    transcript = "\n\n".join(
        f"--- {branch.removeprefix('ask::')} ---\n{answer}"
        for branch, answer in state["branch_results"].items()
    )
    chair_prompt = (
        f"A user asked the following question:\n\n{state['question']}\n\n"
        f"Council members answered independently:\n\n{transcript}\n\n"
        "As chair, synthesize a final answer. Note where the council agrees, "
        "call out any real disagreements and why they might differ, then give "
        "your best consensus recommendation. Be concise."
    )
    try:
        response = ollama.chat(model=CHAIR, messages=[{"role": "user", "content": chair_prompt}])
        verdict = response.message.content or "(empty response)"
    except Exception as e:
        verdict = f"(error: {e})"
    print(f"{CHAIR_COLOR}[verdict]{RESET}\n{verdict}\n")
    return {"verdict": verdict}


def build_council_graph() -> Graph:
    graph = Graph()
    graph.add_node("start", lambda state: None)
    for model in MEMBERS:
        graph.add_node(f"ask::{model}", make_ask_node(model))
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry("start")
    graph.add_fan_out("start", [f"ask::{m}" for m in MEMBERS], "synthesize")
    graph.add_edge("synthesize", END)
    return graph


def run_council(question: str) -> str:
    print(f"\nConvening council on: {question}\n")
    graph = build_council_graph()
    final_state = graph.run({"question": question})
    return final_state["verdict"]


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
