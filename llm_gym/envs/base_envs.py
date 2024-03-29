from dataclasses import dataclass
from typing import List, Dict
from abc import ABC, abstractmethod


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
        self.done = False
        self.examples = []
        self.attempts = 0
        self.correct_first_attempt = False
        self.correct = False
        prompt = self.generate_prompt()
        self.prompt = prompt
        self.cur_chat = [{"role": "user", "content": prompt}]

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
                print(self.cur_chat, repr(action), score)
            elif self.attempts + 1 < self.max_attempts:
                print(self.cur_chat, repr(action), score, "retry")
                self.cur_chat = [
                    {
                        "role": "user",
                        "content": self.prompt + "\n\n" + self.generate_hint(),
                    }
                ]
            else:
                self.done = True
                print(self.cur_chat, repr(action), score)
            self.attempts += 1

    def get_stats(self) -> Dict:
        return {
            "correct_first_attempt": self.correct_first_attempt,
            "correct": self.correct,
        }

    @abstractmethod
    def generate_prompt(self) -> str:
        pass

    @abstractmethod
    def generate_hint(self) -> str:
        pass

    @abstractmethod
    def has_final_result(self, action: str) -> bool:
        pass

    @abstractmethod
    def generate_response(self, action: str) -> str:
        pass

    @abstractmethod
    def score_response(self, action: str) -> float:
        pass


class SingleTurnWithHintsEnv(MultiTurnWithHintsEnv):
    def has_final_result(self, action: str) -> bool:
        return True

    def generate_response(self, action: str) -> str:
        raise NotImplementedError()
