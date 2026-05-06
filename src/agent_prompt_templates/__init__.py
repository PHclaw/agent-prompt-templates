"""Prompt Templates for AI Agents."""

from agent_prompt_templates.prompts.cot import ChainOfThought
from agent_prompt_templates.prompts.react import ReAct
from agent_prompt_templates.prompts.sft import SelfAsk
from agent_prompt_templates.prompts.tot import TreeOfThoughts
from agent_prompt_templates.prompts.fewshot import FewShot
from agent_prompt_templates.prompts.system import SystemPrompt

__all__ = [
    "ChainOfThought",
    "ReAct",
    "SelfAsk",
    "TreeOfThoughts",
    "FewShot",
    "SystemPrompt",
]
