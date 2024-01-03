from typing import Dict, List

import random

EXAMPLE = """
Example -- you MUST solve it like this, where digits are computed smallest to largest.

To calculate the product of 38152 and 76803, smallest to largest digit:

38152 * 7 = reversed(460762) = 267064
38152 * 6 = reversed(219822) = 228912
38152 * 8 = reversed(612503) = 305216
38152 * 0 = reversed(0) = 0
38152 * 3 = reversed(654411) = 114456

267064 + 2289120 + 30521600 + 0 + 114456000 = reversed(6560020392) = 2930200656

Therefore, the product of 38152 and 76803 is 2930200656. 
"""

"""
Let's calculate 908 * 844 using the long multiplication method:\n\n

908 * 1 = reversed(10101) = 10101\n\n
844 * 8 = reversed(301159) = 1591150\n\n
844 * 4 = reversed(411010) = 10101011\n\n
844 * 0 = reversed(0) = 0\n\n
844 * 8 = reversed(589002) = 2002958\n\
    
    n0 + 1 + 1 + 1010 = 1111\n\n
    
    Therefore, the product of 908 and 844 is 11110120
"""


class MathBasicEnv:
    def __init__(
        self,
    ):
        pass

    def reset(self):
        self.a = random.randint(0, 100_000)
        self.b = random.randint(0, 100)
        self.op = random.choice(["*"])
        self.start_prompt = f"What is {self.a} {self.op} {self.b}?"
        self.cur_prompt = self.start_prompt
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
        self.steps = 0
        self.examples = []

    def observe(self) -> List[Dict]:
        return [{"role": "user", "content": self.cur_prompt}]

    def step(self, action: str):
        assert not self.done

        correct = str(self.answer) in action.replace(",", "")

        if self.steps == 0:
            self.first_correct = correct
            if not correct:
                if self.op == "*":
                    self.cur_prompt = (
                        self.start_prompt
                        + " Use the long multiplication method."
                        + EXAMPLE
                    )
                else:
                    self.cur_prompt = (
                        self.start_prompt
                        + " Show each step of the process and partial results."
                    )
                self.attempt_rewards.append(-1.0)
            else:
                self.examples.append(
                    {
                        "chat": [
                            {"role": "user", "content": self.start_prompt},
                            {"role": "assistant", "content": action},
                        ],
                        "reward": 1.0,
                    }
                )
                self.attempt_rewards.append(1.0)
                self.done = True
                print(
                    "Correct!",
                    repr(
                        [
                            {"role": "user", "content": self.cur_prompt},
                            {"role": "assistant", "content": action},
                        ]
                    ),
                )
        else:
            self.second_correct = correct
            if correct:
                self.examples.append(
                    {
                        "chat": [
                            {"role": "user", "content": self.start_prompt},
                            {"role": "assistant", "content": action},
                        ],
                        "reward": 1.0,
                    }
                )
                print(
                    "Correct on hint!",
                    repr(
                        [
                            {"role": "user", "content": self.cur_prompt},
                            {"role": "assistant", "content": action},
                        ]
                    ),
                )
                self.attempt_rewards.append(1.0)
            else:
                print(
                    "NOT CORRECT on hint!",
                    repr(
                        [
                            {"role": "user", "content": self.cur_prompt},
                            {"role": "assistant", "content": action},
                        ]
                    ),
                )
                self.attempt_rewards.append(-1.0)
            self.done = True

        self.steps += 1

    def to_examples(self) -> List[Dict]:
        return self.examples
