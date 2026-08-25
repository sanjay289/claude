#!/usr/bin/env python3
"""hermes_agent.py, ported from the hand-rolled graph_engine.py onto real
LangGraph. Everything except the graph itself is imported unchanged from
hermes_agent.py (TOOLS, TOOL_MAP, call_tool, looks_like_fake_tool_call,
skills) — this file only replaces build_agent_graph/agent_loop.

The graph shape is identical to graph_engine's: check_iterations ->
call_model -> (execute_tools | retry_fake | finish) -> loop back, gated by
an iteration cap. One real semantic difference forced a restructure:
graph_engine's router functions run on the *same* mutable state dict as
nodes, so `check_max_iterations`/`route_after_model` could set
`state["answer"]` as a side effect while also picking the next node.
LangGraph routers only return a routing key — they can't write state, only
a *node*'s return value gets merged. So the two places that used to set
`answer` from inside a router are now the tiny terminal nodes
`finish`/`finish_max_iterations` below, each reached via the router's
mapping instead of going straight to END.

Run: python3 hermes_agent_langgraph.py
"""

from __future__ import annotations

import json
from typing import TypedDict

import ollama
from langgraph.graph import END, StateGraph

from hermes_agent import (
    MAX_FAKE_TOOL_CALL_RETRIES,
    MODEL,
    TOOLS,
    call_tool,
    looks_like_fake_tool_call,
)


class State(TypedDict):
    messages: list
    iteration: int
    fake_tool_call_retries: int
    last_content: str | None
    last_tool_calls: list | None
    answer: str | None


def build_agent_graph(max_iterations: int):
    def check_iterations(state: State) -> dict:
        return {}

    def call_model(state: State) -> dict:
        iteration = state["iteration"] + 1
        print(f"\n[iteration {iteration}]")
        response = ollama.chat(model=MODEL, messages=state["messages"], tools=TOOLS)
        msg = response.message
        messages = state["messages"] + [{
            "role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls
        }]
        return {
            "iteration": iteration,
            "messages": messages,
            "last_content": msg.content,
            "last_tool_calls": msg.tool_calls,
        }

    def route_after_model(state: State) -> str:
        if state["last_tool_calls"]:
            return "tools"
        if looks_like_fake_tool_call(state["last_content"]):
            if state["fake_tool_call_retries"] < MAX_FAKE_TOOL_CALL_RETRIES:
                return "retry"
            print(f"  [warn] still getting fake tool calls after "
                  f"{MAX_FAKE_TOOL_CALL_RETRIES} retries, giving up and "
                  f"returning raw text")
        return "end"

    def finish(state: State) -> dict:
        return {"answer": state["last_content"] or "(no response)"}

    def retry_fake(state: State) -> dict:
        retries = state["fake_tool_call_retries"] + 1
        print(f"  [warn] tool call written as text, asking model to retry "
              f"({retries}/{MAX_FAKE_TOOL_CALL_RETRIES})")
        messages = state["messages"] + [{
            "role": "user",
            "content": (
                "That was a tool call written as text instead of an actual tool "
                "call. Call the tool for real using the tool-calling interface."
            )
        }]
        return {"fake_tool_call_retries": retries, "messages": messages}

    def execute_tools(state: State) -> dict:
        messages = list(state["messages"])
        for tc in state["last_tool_calls"]:
            name = tc.function.name
            args = (tc.function.arguments if isinstance(tc.function.arguments, dict)
                    else json.loads(tc.function.arguments))
            result = call_tool(name, args)
            messages.append({"role": "tool", "content": result, "name": name})
        return {"messages": messages}

    def check_max_iterations(state: State) -> str:
        if state["iteration"] >= max_iterations:
            return "end"
        return "continue"

    def finish_max_iterations(state: State) -> dict:
        return {"answer": "Max iterations reached."}

    builder = StateGraph(State)
    builder.add_node("check_iterations", check_iterations)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_tools", execute_tools)
    builder.add_node("retry_fake", retry_fake)
    builder.add_node("finish", finish)
    builder.add_node("finish_max_iterations", finish_max_iterations)

    builder.add_edge("__start__", "check_iterations")
    builder.add_conditional_edges("check_iterations", check_max_iterations, {
        "continue": "call_model",
        "end": "finish_max_iterations",
    })
    builder.add_conditional_edges("call_model", route_after_model, {
        "tools": "execute_tools",
        "retry": "retry_fake",
        "end": "finish",
    })
    builder.add_edge("execute_tools", "check_iterations")
    builder.add_edge("retry_fake", "check_iterations")
    builder.add_edge("finish", END)
    builder.add_edge("finish_max_iterations", END)

    return builder.compile()


def agent_loop(user_input: str, max_iterations: int = 10) -> str:
    state: State = {
        "messages": [
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
        ],
        "iteration": 0,
        "fake_tool_call_retries": 0,
        "last_content": None,
        "last_tool_calls": None,
        "answer": None,
    }
    app = build_agent_graph(max_iterations)
    final_state = app.invoke(state)
    return final_state["answer"]


def main():
    print(f"Hermes Agent (LangGraph) — model: {MODEL}")
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
