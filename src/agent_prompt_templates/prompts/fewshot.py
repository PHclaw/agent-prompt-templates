"""Few-shot prompting with examples."""

from dataclasses import dataclass, field

from .base import BasePrompt, PromptContext


@dataclass
class FewShot(BasePrompt):
    """
    Few-shot prompting — provide examples to teach the pattern.
    
    Usage:
        from agent_prompt_templates import FewShot
        
        prompt = FewShot(
            examples=[
                {"input": "2 + 2", "output": "4"},
                {"input": "3 + 5", "output": "8"},
            ],
            example_style="input_output"  # or "qa", "demonstration"
        )
        messages = prompt.build(PromptContext(
            question="5 + 7"
        ))
    """

    name = "few-shot"
    description = "Teach via examples"

    examples: list[dict[str, str]] = field(default_factory=list)
    example_style: str = "input_output"  # "input_output", "qa", "demonstration"
    max_examples: int = 5
    include_label: bool = True

    def _format_example(self, ex: dict[str, str]) -> str:
        style = self.example_style
        if style == "input_output":
            return f'Input: {ex.get("input", "")}\nOutput: {ex.get("output", "")}'
        elif style == "qa":
            return f'Q: {ex.get("question", "")}\nA: {ex.get("answer", "")}'
        elif style == "demonstration":
            return ex.get("demonstration", str(ex))
        else:
            return str(ex)

    def system_message(self, context: PromptContext) -> str:
        return ""

    def user_message(self, context: PromptContext) -> str:
        parts = []

        # Add examples
        examples = self.examples[: self.max_examples]
        if examples:
            if self.include_label:
                parts.append("Examples:")
            for ex in examples:
                parts.append(self._format_example(ex))
            parts.append("")

        # Add the actual question
        if self.include_label:
            parts.append(f"Now you try:\n")
        parts.append(context.question)

        return "\n".join(parts)

    def build(self, context: PromptContext) -> list[dict[str, str]]:
        return [{"role": "user", "content": self.user_message(context)}]

    def add_example(self, input_: str, output: str) -> None:
        """Add an example at runtime."""
        self.examples.append({"input": input_, "output": output})


@dataclass
class FewShotWithSystem(FewShot):
    """Few-shot with a system prompt."""

    system_prompt: str = "You are a helpful assistant."

    def system_message(self, context: PromptContext) -> str:
        return self.system_prompt

    def build(self, context: PromptContext) -> list[dict[str, str]]:
        messages = []
        if self.system_message(context):
            messages.append({"role": "system", "content": self.system_message(context)})
        messages.append({"role": "user", "content": self.user_message(context)})
        return messages
