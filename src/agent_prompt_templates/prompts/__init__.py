"""Prompts package."""

from agent_prompt_templates.prompts.cot import ChainOfThought, ZeroShotCoT
from agent_prompt_templates.prompts.react import ReAct, ReActXML
from agent_prompt_templates.prompts.sft import SelfAsk, SelfConsistency
from agent_prompt_templates.prompts.tot import TreeOfThoughts
from agent_prompt_templates.prompts.fewshot import FewShot, FewShotWithSystem
from agent_prompt_templates.prompts.system import (
    SystemPrompt,
    CodeReviewer,
    DataAnalyst,
    Teacher,
)

__all__ = [
    "ChainOfThought",
    "ZeroShotCoT",
    "ReAct",
    "ReActXML",
    "SelfAsk",
    "SelfConsistency",
    "TreeOfThoughts",
    "FewShot",
    "FewShotWithSystem",
    "SystemPrompt",
    "CodeReviewer",
    "DataAnalyst",
    "Teacher",
]
