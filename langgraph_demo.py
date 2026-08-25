#!/usr/bin/env python3
"""LangGraph version of deep_agent_demo.py — same task (compare LRU/LFU/
write-through cache eviction), same shape (plan + sub-agent delegation +
memory), but built on langgraph instead of a hand-rolled loop. Comments
call out what LangGraph gives you for free vs. what deep_agent_demo.py
hand-rolls.

Run: python3 langgraph_demo.py
"""

import os
import pickle
import sqlite3
import threading
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple, get_checkpoint_id
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

MODEL = "llama3.2:3b"
llm = ChatOllama(model=MODEL, temperature=0)

STRATEGIES = ["LRU", "LFU", "write-through"]


class SimpleSqliteSaver(BaseCheckpointSaver):
    """Minimal SQLite-backed checkpointer, sync-only.

    The published langgraph-checkpoint-sqlite package can't run on this
    device: its only version without a `sqlite-vec` dependency (which has no
    wheel for Android/Termux and no sdist to build) predates a serializer
    method (`JsonPlusSerializer.dumps`) that was later removed from the
    checkpoint core `langgraph` 1.2.11 requires. So this reimplements just
    enough of BaseCheckpointSaver to prove state survives a process restart.

    Simplifications vs. the real InMemorySaver/SqliteSaver: each checkpoint
    is stored whole (channel_values included) rather than split into
    per-channel version blobs, so there's no delta history or partial-update
    sharing across checkpoints — fine for this demo's single-checkpoint-per-
    thread use, wrong for a high-frequency production workload.
    """

    def __init__(self, path: str):
        super().__init__()
        # check_same_thread=False plus this lock: LangGraph's Pregel loop
        # calls the checkpointer from a thread pool (even for a sync
        # .invoke()), and one sqlite3.Connection can't be used
        # concurrently from multiple threads without serializing access.
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        with self.lock:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS checkpoints ("
                "thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
                "parent_id TEXT, checkpoint BLOB, metadata BLOB, "
                "PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id))"
            )
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS writes ("
                "thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
                "task_id TEXT, idx INTEGER, channel TEXT, value BLOB, "
                "PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx))"
            )
            self.conn.commit()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.conn.close()

    def _pack(self, obj) -> bytes:
        return pickle.dumps(self.serde.dumps_typed(obj))

    def _unpack(self, blob: bytes):
        return self.serde.loads_typed(pickle.loads(blob))

    def put(self, config, checkpoint, metadata, new_versions) -> dict:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        parent_id = config["configurable"].get("checkpoint_id")
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?, ?, ?, ?)",
                (thread_id, ns, checkpoint["id"], parent_id,
                 self._pack(checkpoint), self._pack(metadata)),
            )
            self.conn.commit()
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(self, config, writes, task_id, task_path="") -> None:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        with self.lock:
            for idx, (channel, value) in enumerate(writes):
                self.conn.execute(
                    "INSERT OR REPLACE INTO writes VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (thread_id, ns, checkpoint_id, task_id, idx, channel, self._pack(value)),
                )
            self.conn.commit()

    def _row_to_tuple(self, thread_id, ns, checkpoint_id, parent_id, checkpoint_blob, metadata_blob):
        with self.lock:
            writes = self.conn.execute(
                "SELECT task_id, channel, value FROM writes "
                "WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=? ORDER BY idx",
                (thread_id, ns, checkpoint_id),
            ).fetchall()
        return CheckpointTuple(
            config={"configurable": {"thread_id": thread_id, "checkpoint_ns": ns,
                                      "checkpoint_id": checkpoint_id}},
            checkpoint=self._unpack(checkpoint_blob),
            metadata=self._unpack(metadata_blob),
            parent_config=({"configurable": {"thread_id": thread_id, "checkpoint_ns": ns,
                                              "checkpoint_id": parent_id}}
                           if parent_id else None),
            pending_writes=[(tid, ch, self._unpack(v)) for tid, ch, v in writes],
        )

    def get_tuple(self, config) -> CheckpointTuple | None:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)
        with self.lock:
            if checkpoint_id:
                row = self.conn.execute(
                    "SELECT checkpoint_id, parent_id, checkpoint, metadata FROM checkpoints "
                    "WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?",
                    (thread_id, ns, checkpoint_id),
                ).fetchone()
            else:
                row = self.conn.execute(
                    "SELECT checkpoint_id, parent_id, checkpoint, metadata FROM checkpoints "
                    "WHERE thread_id=? AND checkpoint_ns=? ORDER BY checkpoint_id DESC LIMIT 1",
                    (thread_id, ns),
                ).fetchone()
        if row is None:
            return None
        cid, parent_id, checkpoint_blob, metadata_blob = row
        return self._row_to_tuple(thread_id, ns, cid, parent_id, checkpoint_blob, metadata_blob)

    def list(self, config, *, filter=None, before=None, limit=None):
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        with self.lock:
            rows = self.conn.execute(
                "SELECT checkpoint_id, parent_id, checkpoint, metadata FROM checkpoints "
                "WHERE thread_id=? AND checkpoint_ns=? ORDER BY checkpoint_id DESC",
                (thread_id, ns),
            ).fetchall()
        count = 0
        for cid, parent_id, checkpoint_blob, metadata_blob in rows:
            if before and get_checkpoint_id(before) and cid >= get_checkpoint_id(before):
                continue
            if limit is not None and count >= limit:
                break
            count += 1
            yield self._row_to_tuple(thread_id, ns, cid, parent_id, checkpoint_blob, metadata_blob)


# --- sub-agent: a compiled graph, invoked as a node -------------------------
# deep_agent_demo.py's spawn_subagent() builds a fresh `messages` list by
# hand and only returns the final string. create_react_agent gives us the
# same "fresh isolated state, summary comes back" shape without writing the
# tool loop ourselves.
subagent = create_react_agent(llm, tools=[])


def research_strategy(strategy: str) -> str:
    """Run the sub-agent on one strategy. Its own reasoning/messages never
    enter the parent's state — only this return value does."""
    result = subagent.invoke({
        "messages": [
            SystemMessage("You are a focused research assistant. Answer in "
                           "3-4 sentences, directly and concisely."),
            HumanMessage(f"Explain the {strategy} cache eviction strategy "
                          f"and when it's a good fit."),
        ]
    })
    return result["messages"][-1].content


# --- parent graph state ------------------------------------------------------
# TODOS was a bare global in deep_agent_demo.py, shared (and clobberable) by
# whatever called write_todos. Here `plan` and `notes` are fields on State,
# threaded automatically through every node — no global needed.
class State(TypedDict):
    messages: Annotated[list, add_messages]
    plan: list[dict]          # [{"task": ..., "status": "pending"|"done"}]
    notes: dict[str, str]     # strategy name -> researched explanation


def init_plan(state: State) -> dict:
    return {"plan": [{"task": s, "status": "pending"} for s in STRATEGIES]}


def delegate(state: State) -> dict:
    """One node per demo call = the parent's plan loop. In deep_agent_demo.py
    the LLM itself decided when to call spawn_subagent/write_memory; here we
    drive the loop directly since the delegation pattern itself is what's
    being taught, not tool-call routing."""
    notes = dict(state["notes"])
    plan = [dict(p) for p in state["plan"]]
    for p in plan:
        if p["status"] == "pending":
            print(f"  [delegate] sub-agent researching: {p['task']}")
            notes[p["task"]] = research_strategy(p["task"])
            p["status"] = "done"
    return {"plan": plan, "notes": notes}


def summarize(state: State) -> dict:
    combined = "\n\n".join(f"### {k}\n{v}" for k, v in state["notes"].items())
    prompt = (f"Given these three cache eviction strategy explanations:\n\n{combined}\n\n"
              "Write a short comparison note saying which strategy fits which "
              "situation.")
    response = llm.invoke([HumanMessage(prompt)])
    return {"messages": [response]}


builder = StateGraph(State)
builder.add_node("init_plan", init_plan)
builder.add_node("delegate", delegate)
builder.add_node("summarize", summarize)
builder.add_edge("__start__", "init_plan")
builder.add_edge("init_plan", "delegate")
builder.add_edge("delegate", "summarize")
builder.add_edge("summarize", END)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "langgraph_checkpoints.sqlite")


def main():
    print(f"LangGraph deep agent — model: {MODEL}")
    print(f"Task: compare {', '.join(STRATEGIES)} cache eviction strategies\n")

    config = {"configurable": {"thread_id": "cache-comparison"}}

    # Checkpointer = deep_agent_demo.py's write_memory/read_memory, but
    # automatic: every node's state update is persisted to DB_PATH under
    # `thread_id`, no manual file I/O. Holds a real sqlite3 connection, so
    # it's used as a context manager. On a normal (non-Termux) machine this
    # would just be `from langgraph.checkpoint.sqlite import SqliteSaver`.
    with SimpleSqliteSaver(DB_PATH) as saver:
        app = builder.compile(checkpointer=saver)

        existing = app.get_state(config)
        if existing.values.get("notes"):
            print(f"Found existing checkpoint at {DB_PATH} — reusing it "
                  "instead of re-running the sub-agents.\n")
            result = existing.values
        else:
            print("No prior checkpoint found — running the full graph.\n")
            result = app.invoke({"messages": [], "plan": [], "notes": {}}, config=config)

        print(f"\nFinal comparison:\n{result['messages'][-1].content}\n")
        print(f"Plan at end: {result['plan']}")

        # Prove the checkpointer actually persisted state under this
        # thread_id — the equivalent of deep_agent_demo.py listing
        # agent_memory/ at the end, but backed by DB_PATH on disk.
        saved = app.get_state(config)
        print(f"Checkpointed notes keys: {list(saved.values['notes'].keys())}")
        print(f"Checkpoint DB: {DB_PATH}")


if __name__ == "__main__":
    main()
