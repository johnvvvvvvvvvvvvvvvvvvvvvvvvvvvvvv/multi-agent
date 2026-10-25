"""
Minimal multi-agent system: orchestrator -> parallel workers -> synthesizer.

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY=your-key
    python multi_agent.py "Research the pros and cons of remote work"
"""
import asyncio
import json
import re
import sys

from anthropic import AsyncAnthropic

client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from the environment
MODEL = "claude-sonnet-5-5"


class Agent:
    """An agent is just a name + a system prompt + an LLM call."""

    def __init__(self, name: str, system: str, max_tokens: int = 1500):
        self.name = name
        self.system = system
        self.max_tokens = max_tokens

    async def run(self, task: str) -> str:
        resp = await client.messages.create(
            model=MODEL,
            max_tokens=self.max_tokens,
            system=self.system,
            messages=[{"role": "user", "content": task}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")


orchestrator = Agent(
    "orchestrator",
    "You break a goal into 2-4 independent subtasks for specialist agents. "
    "Reply with ONLY a JSON list of objects: "
    '[{"role": "<specialist role>", "task": "<clear instruction>"}]',
)

synthesizer = Agent(
    "synthesizer",
    "You merge the outputs of several specialists into one coherent, "
    "well-structured final answer. Resolve contradictions and remove duplication.",
    max_tokens=2500,
)


def parse_plan(text: str) -> list[dict]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"Orchestrator did not return JSON:\n{text}")
    return json.loads(match.group(0))


async def run_worker(role: str, task: str) -> tuple[str, str]:
    worker = Agent(role, f"You are an expert {role}. Be concise and concrete.")
    result = await worker.run(task)
    print(f"  [done] {role}")
    return role, result


async def main(goal: str) -> None:
    print("Planning...")
    plan = parse_plan(await orchestrator.run(goal))
    for step in plan:
        print(f"  - {step['role']}: {step['task']}")

    print("Running workers in parallel...")
    results = await asyncio.gather(
        *(run_worker(step["role"], step["task"]) for step in plan)
    )

    print("Synthesizing...")
    combined = "\n\n".join(f"## {role}\n{output}" for role, output in results)
    final = await synthesizer.run(f"Goal: {goal}\n\nSpecialist outputs:\n{combined}")

    print("\n=== FINAL ANSWER ===\n")
    print(final)


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "Plan a launch for a new productivity app"))
