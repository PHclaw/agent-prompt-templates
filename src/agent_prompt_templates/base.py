"""Base class for all prompt templates."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PromptContext:
    """Context passed to templates."""
    question: str
    examples: list[dict[str, str]] = field(default_factory=list)
    system: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "examples": self.examples,
            "system": self.system,
            **self.extra,
        }


class BasePrompt(ABC):
    """Base class for all prompt templates."""

    name: str
    description: str

    @abstractmethod
    def build(self, context: PromptContext) -> list[dict[str, str]]:
        """
        Build messages from context.
        Returns list of message dicts with 'role' and 'content'.
        """
        ...

    def render(self, context: PromptContext) -> str:
        """Render as a single string (for non-chat models)."""
        messages = self.build(context)
        return "\n\n".join(m["content"] for m in messages)

    def system_message(self, context: PromptContext) -> str:
        """Build system message."""
        return ""

    def user_message(self, context: PromptContext) -> str:
        """Build the main user message."""
        return context.question
