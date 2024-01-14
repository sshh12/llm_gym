import re
from typing import Dict
from functools import lru_cache

from datasets import load_dataset

from llm_gym.envs.env_utils import (
    run_python_code_unsafe,
    get_openai_structured_response,
)
from llm_gym.envs.base_envs import MultiTurnWithHintsEnv, EnvExample


PYTHON_PREFIX = """You are an expert assistant that can run python to help answer questions.

Use this format for running code if needed and wait for the user to respond with the results:
```python
# your code
```"""


class MetaMathQuestionLoader:
    def __init__(self):
        self.data = load_dataset("meta-math/MetaMathQA", split="train").shuffle()
        self.idx = 0

    def get_question(self) -> Dict:
        row = self.data[self.idx]
        self.idx += 1
        return row


@lru_cache(maxsize=1)
def get_question_loader():
    return MetaMathQuestionLoader()


class PythonMetaMathGPTEvalHintsEnv(MultiTurnWithHintsEnv):
    def generate_prompt(self) -> str:
        loader = get_question_loader()
        question = loader.get_question()
        prompt = PYTHON_PREFIX + f"\n\n{question['query']}?"
        self.answer = question["response"]
        self.hint = "Hint: Think step by step"
        # self.examples.append(
        #     EnvExample(
        #         chat=[
        #             {"role": "user", "content": prompt},
        #             {"role": "assistant", "content": self.answer},
        #         ],
        #         reward=1.0,
        #     )
        # )
        return prompt

    def generate_hint(self) -> str:
        return self.hint

    def has_final_result(self, action: str) -> bool:
        has_code_block = (
            len(re.findall(r"```python\n(.*?)\n```", action, re.DOTALL)) > 0
        )
        return not has_code_block or len(self.cur_chat) > 2

    def generate_response(self, action: str) -> str:
        code = re.findall(r"```python\n(.*?)\n```", action, re.DOTALL)[0]
        out = run_python_code_unsafe(code)
        return f"output:\n```{out}```"

    def score_response(self, action: str) -> float:
        prompt = f"You are evaluating an assistants response to a question. \n\nAssistant Answer: {action}\n\nCorrect Answer: {self.answer}.\nDid the assistant get the answer correct?"
        resp = get_openai_structured_response(
            prompt,
            {
                "name": "score_response",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "is_correct": {
                            "type": "boolean",
                            "description": "Are the responses roughly the same?",
                        },
                        "hint": {
                            "type": "string",
                            "description": "Provide a useful hint to the assistant.",
                        },
                    },
                    "required": ["is_correct"],
                },
            },
        )
        if "hint" in resp:
            self.hint = "Hint: " + resp["hint"]
        return float(resp["is_correct"])
