from dataclasses import dataclass
from typing import List, Dict
from abc import ABC


@dataclass
class EnvExample:
    chat: List[Dict]
    reward: float


class BaseEnv(ABC):
    pass


class MultiTurnWithHintsEnv(BaseEnv):
    def __init__(self, max_attempts: int = 2):
        self.max_attempts = max_attempts

    def reset(self):
        prompt = self.generate_prompt()
        self.prompt = prompt
        self.cur_chat = [{"role": "user", "content": prompt}]
        self.done = False
        self.examples = []
        self.attempts = 0
        self.correct_first_attempt = False
        self.correct = False

    def observe(self) -> List[Dict]:
        return self.cur_chat

    def to_examples(self) -> List[EnvExample]:
        return self.examples

    def is_done(self) -> bool:
        return self.done

    def is_ready(self) -> bool:
        return True

    def reward_first_attempt(self) -> float:
        return float(self.correct_first_attempt)

    def reward_best(self) -> float:
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
                self.examples.append(EnvExample(chat=chat_example, reward=score))
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
