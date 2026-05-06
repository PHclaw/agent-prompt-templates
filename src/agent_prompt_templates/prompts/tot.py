"""Tree of Thoughts — explore multiple reasoning branches."""

from dataclasses import dataclass, field

from .base import BasePrompt, PromptContext


@dataclass
class TreeOfThoughts(BasePrompt):
    """
    Tree of Thoughts — explore multiple reasoning paths in parallel.
    Evaluate and prune branches based on promisingness.
    
    Usage:
        from agent_prompt_templates import TreeOfThoughts
        
        prompt = TreeOfThoughts(
            n_branches=3,
            max_depth=4,
            evaluation_criteria="Does this reasoning lead to a correct answer?"
        )
    """

    name = "tree-of-thoughts"
    description = "Explore multiple reasoning branches and pick the best"

    n_branches: int = 3
    max_depth: int = 4
    evaluation_criteria: str = "Does this reasoning path seem promising?"
    include_evaluation: bool = True

    SYSTEM_PROMPT = """You are a strategic problem solver using Tree of Thoughts.
For complex problems, explore multiple approaches in parallel.

Process:
1. Generate several possible reasoning paths
2. Evaluate each path's promisingness
3. Pursue the most promising branches
4. Continue until you find a good solution

Be explicit about:
- What each branch is trying
- Why you evaluate it as promising/unpromising
- When you prune a branch and why"""

    def system_message(self, context: PromptContext) -> str:
        parts = [self.SYSTEM_PROMPT]
        if self.n_branches:
            parts.append(f"\nExplore {self.n_branches} branches in parallel.")
        if self.max_depth:
            parts.append(f"Maximum depth: {self.max_depth} steps.")
        if self.evaluation_criteria:
            parts.append(f"\nEvaluation criteria: {self.evaluation_criteria}")
        return "".join(parts)

    def user_message(self, context: PromptContext) -> str:
        return context.question

    def build(self, context: PromptContext) -> list[dict[str, str]]:
        messages = []
        if self.system_message(context):
            messages.append({"role": "system", "content": self.system_message(context)})
        messages.append({"role": "user", "content": self.user_message(context)})
        return messages
