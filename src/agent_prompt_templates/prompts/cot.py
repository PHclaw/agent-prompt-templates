"""Chain of Thought (CoT) prompting."""

from dataclasses import dataclass

from .base import BasePrompt, PromptContext


@dataclass
class ChainOfThought(BasePrompt):
    """
    Chain of Thought prompting — decompose complex problems step by step.
    
    Usage:
        from agent_prompt_templates import ChainOfThought
        
        prompt = ChainOfThought()
        messages = prompt.build(PromptContext(
            question="If a train travels 120km in 2 hours, what is its speed?"
        ))
    """

    name = "chain-of-thought"
    description = "Step-by-step reasoning for complex problems"

    include_verification: bool = True
    max_steps: int = 10

    SYSTEM_PROMPT = """You are a careful thinker. Think step by step before answering.
Break down complex problems into smaller, manageable steps.
Show your reasoning process clearly."""

    def system_message(self, context: PromptContext) -> str:
        parts = [self.SYSTEM_PROMPT]
        if self.include_verification:
            parts.append("\nAfter reaching your conclusion, verify it makes sense.")
        if self.max_steps:
            parts.append(f"\nAim for {self.max_steps} or fewer reasoning steps.")
        return "".join(parts)

    def user_message(self, context: PromptContext) -> str:
        return (
            f"{context.question}\n\n"
            "Think step by step and show your reasoning."
        )

    def build(self, context: PromptContext) -> list[dict[str, str]]:
        messages = []
        if self.system_message(context):
            messages.append({"role": "system", "content": self.system_message(context)})
        messages.append({"role": "user", "content": self.user_message(context)})
        return messages


@dataclass
class ZeroShotCoT(BasePrompt):
    """
    Zero-shot Chain of Thought — just add 'think step by step'.
    Minimal but surprisingly effective.
    """

    name = "zero-shot-cot"
    description = "Minimal CoT — just add a trigger phrase"

    trigger = "Think step by step."

    def system_message(self, context: PromptContext) -> str:
        return ""

    def user_message(self, context: PromptContext) -> str:
        return f"{context.question}\n\n{self.trigger}"

    def build(self, context: PromptContext) -> list[dict[str, str]]:
        return [{"role": "user", "content": self.user_message(context)}]
