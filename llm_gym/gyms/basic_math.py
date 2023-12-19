import random
import torch


class BasicMathGym:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def reset(self):
        self.a = random.randint(0, 100_000)
        self.b = random.randint(0, 100_000)
        self.prompt = f"What is {self.a} + {self.b}?"
        self.state = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": self.prompt},
            ],
            return_tensors="pt",
            padding=True,
            max_length=25,
        )[0]
        self.done = False
        self.correct = False
        self.steps = 0
        return self.state

    def step(self, action: int):
        reward = 0
        if not self.done and action == self.tokenizer.eos_token_id:
            resp = self.tokenizer.decode(self.state)
            correct = str(self.a + self.b) in resp.replace(",", "")
            print("done!", resp, "CORRECT" if correct else "INCORRECT")
            reward = 1.0 if correct else -1.0
            self.correct = correct
            self.done = True
        elif self.steps > 70:
            self.done = True
            self.correct = False
            print("done failed!", self.tokenizer.decode(self.state))
            reward = -1.0
        if not self.done:
            self.state = torch.concat([self.state, torch.IntTensor([action])])
            self.steps += 1
        else:
            self.state = torch.concat(
                [self.state, torch.IntTensor([self.tokenizer.eos_token_id])]
            )
        return self.state, reward, int(self.done)
