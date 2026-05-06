"""Common system prompt templates."""

from dataclasses import dataclass, field

from .base import BasePrompt, PromptContext


@dataclass
class SystemPrompt(BasePrompt):
    """
    System prompt builder — common roles and boilerplate.
    
    Usage:
        from agent_prompt_templates import SystemPrompt
        
        prompt = SystemPrompt(
            role="assistant",
            persona="helpful, concise, and accurate",
            constraints=["Think before answering", "Admit when unsure"]
        )
    """

    name = "system-prompt"
    description = "Common system prompt roles and boilerplate"

    role: str = "assistant"
    persona: str = "helpful, concise, and accurate"
    constraints: list[str] = field(default_factory=list)
    output_format: str = ""  # e.g., "Respond in JSON format."

    def system_message(self, context: PromptContext) -> str:
        parts = []
        
        if self.role:
            parts.append(f"You are a {self.role}.")
        
        if self.persona:
            parts.append(f"Be {self.persona}.")
        
        if self.constraints:
            parts.append("\nGuidelines:")
            for c in self.constraints:
                parts.append(f"- {c}")
        
        if self.output_format:
            parts.append(f"\n{self.output_format}")
        
        if context.system:
            parts.append(f"\n{context.system}")
        
        return "\n".join(parts)

    def user_message(self, context: PromptContext) -> str:
        return context.question

    def build(self, context: PromptContext) -> list[dict[str, str]]:
        messages = []
        if self.system_message(context):
            messages.append({"role": "system", "content": self.system_message(context)})
        messages.append({"role": "user", "content": self.user_message(context)})
        return messages


@dataclass
class CodeReviewer(SystemPrompt):
    """Code review specific system prompt."""

    name = "code-reviewer"

    def __post_init__(self):
        self.role = "code reviewer"
        self.persona = "thorough, constructive, and precise"
        self.constraints = [
            "Review for correctness, performance, and readability",
            "Suggest concrete improvements with code examples",
            "Flag potential bugs and security issues",
            "Be specific about what to change and why",
        ]


@dataclass
class DataAnalyst(SystemPrompt):
    """Data analysis specific system prompt."""

    name = "data-analyst"

    def __post_init__(self):
        self.role = "data analyst"
        self.persona = "rigorous, quantitative, and clear"
        self.constraints = [
            "Base conclusions on data, not assumptions",
            "Explain methodology when analyzing",
            "Use precise numbers and statistics",
            "Acknowledge limitations and uncertainty",
        ]


@dataclass
class Teacher(SystemPrompt):
    """Teaching specific system prompt."""

    name = "teacher"

    def __post_init__(self):
        self.role = "teacher"
        self.persona = "patient, clear, and encouraging"
        self.constraints = [
            "Explain concepts at the right level of detail",
            "Use analogies and examples",
            "Check understanding before moving on",
            "Be encouraging and supportive",
        ]
