import re
from typing import Dict
from functools import lru_cache

from datasets import load_dataset

from llm_gym.envs.env_utils import (
    get_openai_structured_response,
)
from llm_gym.envs.base_envs import SingleTurnWithHintsEnv, EnvExample


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


class MetaMathGPTEvalHintsEnv(SingleTurnWithHintsEnv):
    def generate_prompt(self) -> str:
        loader = get_question_loader()
        question = loader.get_question()
        self.query = question["query"]
        self.answer = question["response"]
        self.hint = "Hint: Show your work"
        prompt = self.query + '\n\nThink step by step.'
        self.examples.append(
            EnvExample(
                chat=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": self.answer},
                ],
                reward=1.0,
            )
        )
        return prompt

    def generate_hint(self) -> str:
        return self.hint

    def score_response(self, action: str) -> float:
        if len(action.strip()) == 0:
            return 0.0
        prompt = f"You are evaluating an assistants response to a question. Question: {self.query}\n\nAssistant Answer: {action}\n\nCorrect Answer: {self.answer}.\nDid the assistant get the answer correct?"
        try:
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
        except Exception:
            return 0.0
