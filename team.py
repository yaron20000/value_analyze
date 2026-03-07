"""
Agent team for value_analyze project.

Three specialized agents orchestrated by a main loop:
  - product   : reads todo/docs, writes a clear task spec
  - developer : implements the spec by editing code
  - reviewer  : reviews changes and reports issues

Usage:
    python team.py "finish the account feature"
    python team.py  # runs the full todo list
"""

import asyncio
import sys
from pathlib import Path

from claude_agent_sdk import (
    query,
    AgentDefinition,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    AssistantMessage,
    ToolUseBlock,
    TextBlock,
    ThinkingBlock,
)

PROJECT_DIR = str(Path(__file__).parent)

# ── Agent definitions ─────────────────────────────────────────────────────────

PRODUCT_AGENT = AgentDefinition(
    description=(
        "Product manager for the value_analyze Borsdata project. "
        "Reads the codebase, README, and todo list to produce a clear, "
        "actionable specification for a single task."
    ),
    prompt=(
        "You are a product manager for a Python/PostgreSQL stock-data pipeline "
        "that fetches data from the Borsdata API and stores it for ML analysis.\n\n"
        "Your job:\n"
        "1. Read README.md, todo.md, and relevant source files to understand the project.\n"
        "2. For the requested task, write a concise spec:\n"
        "   - What needs to be built or fixed\n"
        "   - Acceptance criteria\n"
        "   - Files likely to be changed\n"
        "   - Any known constraints or gotchas\n"
        "Output only the spec — no implementation."
    ),
    tools=["Read", "Glob", "Grep"],
)

DEVELOPER_AGENT = AgentDefinition(
    description=(
        "Senior Python developer for the value_analyze Borsdata project. "
        "Receives a task spec and implements it by editing or creating files."
    ),
    prompt=(
        "You are a senior Python developer working on a Borsdata API data pipeline "
        "that stores stock market data in PostgreSQL for ML processing.\n\n"
        "Your job:\n"
        "1. Read the task spec provided to you.\n"
        "2. Explore the codebase as needed (Read, Glob, Grep).\n"
        "3. Implement the changes using Edit and Write tools.\n"
        "4. Follow existing code style and patterns.\n"
        "5. Do not add unnecessary complexity — minimal, focused changes only.\n"
        "6. Report every file you changed and a one-line reason for each change."
    ),
    tools=["Read", "Glob", "Grep", "Edit", "Write", "Bash"],
)

REVIEWER_AGENT = AgentDefinition(
    description=(
        "Code reviewer for the value_analyze Borsdata project. "
        "Reviews recent changes for correctness, security, and code quality."
    ),
    prompt=(
        "You are a senior code reviewer for a Python/PostgreSQL stock-data pipeline.\n\n"
        "Your job:\n"
        "1. Read the changed files reported by the developer.\n"
        "2. Check for:\n"
        "   - Correctness (does it match the spec?)\n"
        "   - SQL injection, secrets in code, or other security issues\n"
        "   - Broken imports or missing dependencies\n"
        "   - Style inconsistencies with the rest of the codebase\n"
        "3. Output a structured review:\n"
        "   - APPROVED / NEEDS CHANGES\n"
        "   - List of issues (file:line — description)\n"
        "   - Suggestions (optional)\n"
        "Do not modify files — review only."
    ),
    tools=["Read", "Glob", "Grep"],
)

# ── Orchestrator ──────────────────────────────────────────────────────────────

async def run_agent(name: str, prompt: str, agents: dict, session_id: str | None = None) -> tuple[str, str]:
    """Run a single agent and return (result_text, session_id)."""
    print(f"\n{'='*60}")
    print(f"  {name.upper()} AGENT")
    print(f"{'='*60}")

    result_text = ""
    captured_session_id = session_id

    options = ClaudeAgentOptions(
        cwd=PROJECT_DIR,
        allowed_tools=["Read", "Glob", "Grep", "Edit", "Write", "Bash"],
        permission_mode="acceptEdits",
        agents=agents,
        max_turns=30,
    )

    if session_id:
        options.resume = session_id

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, SystemMessage):
            if message.subtype == "init":
                captured_session_id = getattr(message, "session_id", None)
                print(f"  [session: {captured_session_id}]")
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ThinkingBlock):
                    pass  # skip raw thinking output
                elif isinstance(block, ToolUseBlock):
                    tool_input = getattr(block, "input", {})
                    hint = (
                        tool_input.get("file_path")
                        or tool_input.get("pattern")
                        or tool_input.get("query", "")[:60]
                        or ""
                    )
                    print(f"  > {block.name}({hint})")
                elif isinstance(block, TextBlock) and block.text.strip():
                    # Print first 120 chars of any text turn for context
                    preview = block.text.strip().replace("\n", " ")[:120]
                    print(f"  ... {preview}")
        elif isinstance(message, ResultMessage):
            result_text = message.result
            print("\n── RESULT ──")
            print(result_text)

    return result_text, captured_session_id


async def main():
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None

    agents = {
        "product": PRODUCT_AGENT,
        "developer": DEVELOPER_AGENT,
        "reviewer": REVIEWER_AGENT,
    }

    # ── Step 1: Product agent writes the spec ─────────────────────────────────
    product_prompt = (
        f"Task: {task}\n\nRead the project files and produce a clear spec for this task."
        if task
        else (
            "Read todo.md and the project files. "
            "Pick the highest-priority incomplete item and produce a clear spec for it."
        )
    )

    spec, _ = await run_agent("product", product_prompt, agents)

    # ── Step 2: Developer implements the spec ─────────────────────────────────
    dev_prompt = (
        f"Here is the task spec from the product manager:\n\n{spec}\n\n"
        "Implement it now. Report every file you changed."
    )

    dev_result, _ = await run_agent("developer", dev_prompt, agents)

    # ── Step 3: Reviewer checks the changes ───────────────────────────────────
    review_prompt = (
        f"The developer completed this task. Here is their report:\n\n{dev_result}\n\n"
        f"Original spec:\n\n{spec}\n\n"
        "Review the changed files and provide your assessment."
    )

    await run_agent("reviewer", review_prompt, agents)

    print("\n" + "="*60)
    print("  TEAM RUN COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
