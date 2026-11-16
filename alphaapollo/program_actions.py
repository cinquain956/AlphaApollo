"""Shared AlphaApollo XML action prompts for executable geometry programs."""

PROGRAM_ACTION_PROTOCOL = """Now it is your turn to respond to the current step.
First conduct the reasoning process. This process MUST be enclosed within <think> </think> tags.
After completing your reasoning, choose exactly one of the following actions (do not perform both):
1) <python_code>...</python_code>: emit exactly ONE block containing a complete executable Python 3 program. The block must be a full replacement program, never a patch, fragment, SEARCH/REPLACE edit, or prose explanation.
2) <answer>...</answer>: only when finishing an AlphaApollo multi-step trajectory, emit exactly ONE block containing the final complete executable Python 3 program.
The XML action may contain an optional ```python fence inside the selected tag. Do not place Python source outside the selected action tag."""


def build_program_action_prompt(task_instruction: str, *, final_answer: bool = True) -> str:
    """Build the shared protocol around a problem-specific instruction."""
    protocol = PROGRAM_ACTION_PROTOCOL
    if not final_answer:
        protocol = protocol.replace(
            "2) <answer>...</answer>: only when finishing an AlphaApollo multi-step trajectory, emit exactly ONE block containing the final complete executable Python 3 program.\n",
            "",
        ).replace(
            "choose exactly one of the following actions (do not perform both):",
            "emit the following single action:",
        )
        protocol += (
            "\nFor this full-rewrite discovery call, the response MUST start with <think>. "
            "Keep the think block under 80 words, close it, and begin <python_code> within "
            "the first 120 output tokens. Put all implementation detail in the complete "
            "program and always close </python_code>."
        )
    return (
        "You are a program discovery agent tasked with improving an executable solution "
        "while preserving the required interface and deterministic validity constraints.\n\n"
        f"Your task: {task_instruction.strip()}\n\n"
        f"{protocol}"
    )


def build_feedback_instruction() -> str:
    return (
        "Inspect the deterministic checker feedback above. If another step remains, reason in "
        "<think> and submit one complete replacement program in <python_code>. Use <answer> "
        "only for the final complete program."
    )
