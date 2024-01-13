from typing import Dict, List
import re
import random

from llm_gym.envs.env_utils import run_python_code_unsafe


PYTHON_PREFIX = """
<system>
You are an expert assistant that can run python to help answer questions.

```python
...
```
</system>"""


class MultiTurnWithHintsEnv:
    def reset(self):
        prompt = self.generate_prompt()
        self.prompt = prompt
        self.cur_chat = [{"role": "user", "content": prompt}]
        self.done = False
        self.examples = []
        self.max_attempts = 2
        self.attempts = 0
        self.correct_first_attempt = False
        self.correct = False

    def observe(self) -> List[Dict]:
        return self.cur_chat

    def to_examples(self) -> List[Dict]:
        return self.examples

    def is_done(self) -> bool:
        return self.done

    def is_ready(self) -> bool:
        return True

    def reward_first_attempt(self) -> float:
        return float(self.correct_first_attempt)

    def reward(self) -> float:
        return float(self.correct)

    def step(self, action: str):
        print(self.cur_chat, action)
        if not self.has_final_result(action):
            user_resp = self.generate_response(action)
            self.cur_chat.extend(
                [
                    {"role": "assistant", "content": action},
                    {"role": "user", "content": user_resp},
                ]
            )
        else:
            score = self.score_response(action)
            if score > 0.0:
                chat_example = (
                    [{"role": "user", "content": self.prompt}]
                    + self.cur_chat[1:]
                    + [{"role": "assistant", "content": action}]
                )
                self.examples.append(
                    {
                        "chat": chat_example,
                        "reward": score,
                    }
                )
                self.correct = True
                if self.attempts == 0:
                    self.correct_first_attempt = True
                self.done = True
            elif self.attempts + 1 < self.max_attempts:
                self.cur_chat = [
                    {
                        "role": "user",
                        "content": self.prompt + "\n\n" + self.generate_hint(),
                    }
                ]
            else:
                self.done = True
            self.attempts += 1

    def get_stats(self) -> Dict:
        return {
            "correct_first_attempt": self.correct_first_attempt,
            "correct": self.correct,
        }

    def generate_prompt(self) -> str:
        pass

    def generate_hint(self) -> str:
        pass

    def has_final_result(self, action: str) -> bool:
        pass

    def generate_response(self, action: str) -> str:
        pass

    def score_response(self, action: str) -> float:
        pass


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
        has_code_block = len(re.findall(r"```python\n(.*)\n```", action, re.DOTALL)) > 0
        return not has_code_block or len(self.cur_chat) > 2

    def generate_response(self, action: str) -> str:
        code = re.findall(r"```python\n(.*)\n```", action, re.DOTALL)[0]
        out = run_python_code_unsafe(code)
        return out

    def score_response(self, action: str) -> float:
        return float(str(self.answer) in action.replace(",", ""))
