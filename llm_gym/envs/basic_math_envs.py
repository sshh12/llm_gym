import re
import random

from llm_gym.envs.env_utils import run_python_code_unsafe
from llm_gym.envs.base_envs import MultiTurnWithHintsEnv


PYTHON_PREFIX = """You are an expert assistant that can run python to help answer questions.

Use this format for running code if needed and wait for the user to respond with the results:
```python
# your code
```"""


class PythonMathHintsEnv(MultiTurnWithHintsEnv):
    def generate_prompt(self) -> str:
        self.a = random.randint(0, 100_000)
        self.b = random.randint(0, 100_000)
        self.op = random.choice(["*", "+", "-"])
        prompt = PYTHON_PREFIX + f"\n\nWhat is {self.a} {self.op} {self.b}?"
        self.answer = eval(f"{self.a} {self.op} {self.b}")
        return prompt

    def generate_hint(self) -> str:
        return "Hint: Use a python code block to run code and print the results. Wait for the results to be provided by the user before answering. Be sure to print() the results."

    def has_final_result(self, action: str) -> bool:
        has_code_block = (
            len(re.findall(r"```python\n(.*?)\n```", action, re.DOTALL)) > 0
        )
        return (
            not has_code_block
            or len(self.cur_chat) > 2
            or self.answer in repr(self.cur_chat).replace(",", "")
        )

    def generate_response(self, action: str) -> str:
        code = re.findall(r"```python\n(.*?)\n```", action, re.DOTALL)[0]
        out = run_python_code_unsafe(code)
        return f"output:\n```{out}```"

    def score_response(self, action: str) -> float:
        return float(str(self.answer) in action.replace(",", ""))
