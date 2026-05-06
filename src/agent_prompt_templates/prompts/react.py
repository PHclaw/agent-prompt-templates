"""ReAct (Reason + Act) prompting — interleaves reasoning and actions."""

from dataclasses import dataclass, field

from .base import BasePrompt, PromptContext


@dataclass
class ReAct(BasePrompt):
    """
    ReAct = Reason + Act. Interleaves reasoning traces with actions.
    Perfect for tool-use agents.
    
    Each step: Thought → Action → Observation → ...
    
    Usage:
        from agent_prompt_templates import ReAct
        
        prompt = ReAct(
            available_tools=["search(query)", "read_file(path)"],
            max_steps=8
        )
        messages = prompt.build(PromptContext(
            question="What was the weather in Beijing yesterday?"
        ))
    """

    name = "react"
    description = "Reason + Act interleaving for tool-use agents"

    available_tools: list[str] = field(default_factory=list)
    max_steps: int = 8
    include_thought_labels: bool = True

    SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

When you need information, use the available tools. When you have enough
information, give the final answer.

Format each step as:
Thought: what you're thinking
Action: the action to take (if needed)
Observation: the result of the action
... repeat until you can answer ...
Final Answer: your response to the question"""

    def system_message(self, context: PromptContext) -> str:
        parts = [self.SYSTEM_PROMPT]
        if self.available_tools:
            parts.append("\n\nAvailable tools:")
            for tool in self.available_tools:
                parts.append(f"  - {tool}")
        if self.max_steps:
            parts.append(f"\n\nMaximum {self.max_steps} reasoning steps.")
        return "".join(parts)

    def user_message(self, context: PromptContext) -> str:
        return context.question

    def build(self, context: PromptContext) -> list[dict[str, str]]:
        messages = []
        if self.system_message(context):
            messages.append({"role": "system", "content": self.system_message(context)})
        messages.append({"role": "user", "content": self.user_message(context)})
        return messages


@dataclass  
class ReActXML(BasePrompt):
    """
    ReAct with XML-style tool tags — works great with Llama, Qwen, etc.
    """

    name = "react-xml"
    description = "ReAct with XML tool tags"

    available_tools: list[str] = field(default_factory=list)
    max_steps: int = 8

    def system_message(self, context: PromptContext) -> str:
        parts = [
            "You have access to tools. Use them when needed.",
            "Use the following XML format for tool calls:",
            "",
            "<tool_calls>",
            '  <tool name="tool_name">',
            '    <param name="param_name">value</param>',
            "  </tool>",
            "</tool_calls>",
            "",
            "When you need to use a tool, output the XML.",
            "When you have the answer, write: Final Answer: ...",
        ]
        if self.available_tools:
            parts.append("\nAvailable tools:")
            for tool in self.available_tools:
                parts.append(f"  - {tool}")
        return "\n".join(parts)

    def user_message(self, context: PromptContext) -> str:
        return context.question

    def build(self, context: PromptContext) -> list[dict[str, str]]:
        messages = []
        if self.system_message(context):
            messages.append({"role": "system", "content": self.system_message(context)})
        messages.append({"role": "user", "content": self.user_message(context)})
        return messages
