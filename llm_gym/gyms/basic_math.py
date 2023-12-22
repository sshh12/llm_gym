import random
import torch


class BasicMathGym:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def reset(self):
        self.a = random.randint(0, 100_000)
        self.b = random.randint(0, 100_000)
        self.prompt = f"What is {self.a} + {self.b}?"
        self.chat = [
            {"role": "user", "content": self.prompt},
        ]
        self.acc_tokens = []
        self.done = False
        self.correct = False
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        state_chat = list(self.chat)
        tokens = self.tokenizer.apply_chat_template(
            state_chat,
            return_tensors="pt",
        )[0]
        if len(self.acc_tokens) > 0:
            tokens = torch.cat([tokens, torch.IntTensor(self.acc_tokens)])
        return {"input_ids": tokens}

    def step(self, action: int):
        reward = 0.0
        if not self.done:
            if action == self.tokenizer.eos_token_id:
                resp = self.tokenizer.decode(self.acc_tokens)
                correct = str(self.a + self.b) in resp.replace(",", "")
                print("DONE SUCCESS!", resp, "CORRECT" if correct else "INCORRECT")
                reward = 1.0 if correct else -1.0
                self.correct = correct
                self.done = True
            elif self.steps > 100:
                resp = self.tokenizer.decode(self.acc_tokens)
                self.done = True
                self.correct = False
                print("DONE FAIL!", resp)
                reward = -1.0
            self.acc_tokens.append(action)
            self.steps += 1
        return self._get_state(), reward, int(self.done)
