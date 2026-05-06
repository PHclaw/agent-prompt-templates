"""Self-Ask prompting — decompose questions with sub-questions."""

from dataclasses import dataclass

from .base import BasePrompt, PromptContext


@dataclass
class SelfAsk(BasePrompt):
    """
    Self-Ask prompting — the model explicitly asks itself sub-questions
    before reaching a final answer.
    
    Usage:
        from agent_prompt_templates import SelfAsk
        
        prompt = SelfAsk()
        messages = prompt.build(PromptContext(
            question="Who was the father of the current president of the US?"
        ))
    """

    name = "self-ask"
    description = "Explicitly ask sub-questions before answering"

    include_intermediate_answers: bool = True
    max_subquestions: int = 5

    SYSTEM_PROMPT = """You are a helpful assistant. Break complex questions into
simpler sub-questions. Answer each sub-question, then combine them for
the final answer.

Format:
Follow up: [sub-question]
Intermediate answer: [answer to sub-question]
...
So the final answer is: [your response]"""

    def system_message(self, context: PromptContext) -> str:
        return self.SYSTEM_PROMPT

    def user_message(self, context: PromptContext) -> str:
        return context.question

    def build(self, context: PromptContext) -> list[dict[str, str]]:
        messages = []
        if self.system_message(context):
            messages.append({"role": "system", "content": self.system_message(context)})
        messages.append({"role": "user", "content": self.user_message(context)})
        return messages


@dataclass
class SelfConsistency(BasePrompt):
    """
    Self-Consistency — sample multiple reasoning paths and pick the most
    consistent answer. More compute but better accuracy on complex tasks.
    """

    name = "self-consistency"
    description = "Sample multiple paths, pick the most consistent answer"

    n_samples: int = 5
    cot_system: str = "You are a careful thinker. Think step by step."

    def system_message(self, context: PromptContext) -> str:
        return self.cot_system

    def user_message(self, context: PromptContext) -> str:
        return (
            f"{context.question}\n\n"
            "Think step by step. Show your reasoning."
        )

    def build(self, context: PromptContext) -> list[dict[str, str]]:
        messages = []
        if self.system_message(context):
            messages.append({"role": "system", "content": self.system_message(context)})
        messages.append({"role": "user", "content": self.user_message(context)})
        return messages

    def post_process(self, responses: list[str]) -> str:
        """
        Given multiple reasoning paths, pick the most common answer.
        Override this with a voting mechanism.
        """
        from collections import Counter
        # Strip and normalize answers
        answers = []
        for resp in responses:
            # Try to extract answer from response
            lines = resp.strip().split("\n")
            if lines:
                answers.append(lines[-1].strip())
        if not answers:
            return responses[0] if responses else ""
        # Majority vote
        counter = Counter(answers)
        return counter.most_common(1)[0][0]
