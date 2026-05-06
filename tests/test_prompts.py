"""Tests for prompt templates."""

import pytest
from agent_prompt_templates.base import PromptContext
from agent_prompt_templates.prompts.cot import ChainOfThought, ZeroShotCoT
from agent_prompt_templates.prompts.react import ReAct, ReActXML
from agent_prompt_templates.prompts.sft import SelfAsk, SelfConsistency
from agent_prompt_templates.prompts.tot import TreeOfThoughts
from agent_prompt_templates.prompts.fewshot import FewShot, FewShotWithSystem
from agent_prompt_templates.prompts.system import SystemPrompt, CodeReviewer


class TestChainOfThought:
    def test_build(self) -> None:
        prompt = ChainOfThought()
        ctx = PromptContext(question="What is 2+2?")
        messages = prompt.build(ctx)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "2+2" in messages[1]["content"]

    def test_render(self) -> None:
        prompt = ChainOfThought()
        ctx = PromptContext(question="What is 2+2?")
        text = prompt.render(ctx)
        assert "2+2" in text
        assert len(text) > 0


class TestZeroShotCoT:
    def test_no_system_message(self) -> None:
        prompt = ZeroShotCoT()
        ctx = PromptContext(question="What is 2+2?")
        messages = prompt.build(ctx)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "Think step by step" in messages[0]["content"]


class TestReAct:
    def test_with_tools(self) -> None:
        prompt = ReAct(available_tools=["search(q)", "read_file(path)"])
        ctx = PromptContext(question="Find the README")
        messages = prompt.build(ctx)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "search(q)" in messages[0]["content"]
        assert "read_file(path)" in messages[0]["content"]

    def test_without_tools(self) -> None:
        prompt = ReAct()
        ctx = PromptContext(question="Hello")
        messages = prompt.build(ctx)
        assert len(messages) == 2


class TestReActXML:
    def test_xml_format(self) -> None:
        prompt = ReActXML(available_tools=["search(q)"])
        ctx = PromptContext(question="Hello")
        messages = prompt.build(ctx)

        assert messages[0]["role"] == "system"
        assert "<tool_calls>" in messages[0]["content"]


class TestSelfAsk:
    def test_build(self) -> None:
        prompt = SelfAsk()
        ctx = PromptContext(question="Who built the Eiffel Tower?")
        messages = prompt.build(ctx)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Follow up:" in messages[0]["content"]


class TestSelfConsistency:
    def test_build(self) -> None:
        prompt = SelfConsistency()
        ctx = PromptContext(question="What is 2+2?")
        messages = prompt.build(ctx)
        assert len(messages) == 2

    def test_post_process(self) -> None:
        prompt = SelfConsistency()
        responses = ["The answer is 4", "The answer is 4", "The answer is 5"]
        result = prompt.post_process(responses)
        assert result == "The answer is 4"


class TestTreeOfThoughts:
    def test_build(self) -> None:
        prompt = TreeOfThoughts(n_branches=3, max_depth=4)
        ctx = PromptContext(question="Best way to learn Python?")
        messages = prompt.build(ctx)

        assert len(messages) == 2
        assert "3 branches" in messages[0]["content"]
        assert "depth" in messages[0]["content"]


class TestFewShot:
    def test_with_examples(self) -> None:
        prompt = FewShot(
            examples=[
                {"input": "a", "output": "A"},
                {"input": "b", "output": "B"},
            ]
        )
        ctx = PromptContext(question="c")
        messages = prompt.build(ctx)

        assert len(messages) == 1
        assert "Input: a" in messages[0]["content"]
        assert "Output: A" in messages[0]["content"]
        assert "c" in messages[0]["content"]

    def test_add_example(self) -> None:
        prompt = FewShot()
        prompt.add_example("x", "X")
        assert len(prompt.examples) == 1


class TestFewShotWithSystem:
    def test_system_message(self) -> None:
        prompt = FewShotWithSystem(system_prompt="You are smart.")
        ctx = PromptContext(question="Hi")
        messages = prompt.build(ctx)

        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are smart."
        assert messages[1]["role"] == "user"


class TestSystemPrompt:
    def test_basic(self) -> None:
        prompt = SystemPrompt(role="assistant", persona="helpful")
        ctx = PromptContext(question="Hello")
        messages = prompt.build(ctx)

        assert len(messages) == 2
        assert "assistant" in messages[0]["content"]
        assert "helpful" in messages[0]["content"]

    def test_constraints(self) -> None:
        prompt = SystemPrompt(
            role="reviewer",
            constraints=["Be thorough", "Be specific"],
        )
        ctx = PromptContext(question="Review this")
        messages = prompt.build(ctx)

        assert "thorough" in messages[0]["content"]
        assert "specific" in messages[0]["content"]


class TestCodeReviewer:
    def test_build(self) -> None:
        prompt = CodeReviewer()
        ctx = PromptContext(question="def foo(): pass")
        messages = prompt.build(ctx)

        assert messages[0]["role"] == "system"
        assert "code reviewer" in messages[0]["content"]
        assert "correctness" in messages[0]["content"]
