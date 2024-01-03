from typing import Dict, List

import random


PREFIX = (
    """
<system>
You are an expert assistant that can run python to help answer questions.

```python
...
```
</system>

""".strip()
    + "\n"
)


class MathFuncsEnv:
    def __init__(
        self,
    ):
        pass

    def reset(self):
        self.a = random.randint(0, 100_000)
        self.b = random.randint(0, 100_000)
        self.op = random.choice(["*", "+", "-"])
        self.start_prompt = PREFIX + f"What is {self.a} {self.op} {self.b}?"
        self.cur_chat = [{"role": "user", "content": self.start_prompt}]
        if self.op == "+":
            self.answer = self.a + self.b
        elif self.op == "*":
            self.answer = self.a * self.b
        else:
            self.answer = self.a - self.b

        self.done = False
        self.attempt_rewards = []
        self.first_correct = False
        self.second_correct = False
        self.attempts = 0
        self.examples = []

    def observe(self) -> List[Dict]:
        return self.cur_chat

    def step(self, action: str):
        assert not self.done

        correct = str(self.answer) in action.replace(",", "")
        print(self.cur_chat, self.answer)
        self.done = True
        self.first_correct = "python" in action

        # self.examples.append(
        #     {
        #         "chat": [
        #             {"role": "user", "content": self.start_prompt},
        #             {
        #                 "role": "assistant",
        #                 f"content": f"```python\nprint({self.a} {self.op} {self.b})\n```",
        #             },
        #             {"role": "user", "content": str(self.answer)},
        #             {
        #                 "role": "assistant",
        #                 f"content": f"The answer is {self.answer}",
        #             },
        #         ],
        #         "reward": 1.0,
        #     }
        # )
        self.examples.append(
            {
                "chat": [
                    {"role": "user", "content": self.start_prompt},
                    {
                        "role": "assistant",
                        f"content": f"I only code in JAVA",
                    },
                ],
                "reward": 1.0,
            }
        )

        self.attempts += 1

    def to_examples(self) -> List[Dict]:
        return self.examples
