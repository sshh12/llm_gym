from typing import Dict, List
import re

from transformers import AutoTokenizer, MistralForCausalLM, LlamaTokenizer
import torch

from llm_gym.constants import IGNORE_INDEX
from llm_gym.model_utils import (
    fix_tokenizer,
    load_model_qlora,
    load_lora_model,
    left_pad,
)
from llm_gym.agents.chat_agent import BaseChatAgent


class MistralChatAgent(BaseChatAgent):
    def __init__(
        self,
        model: MistralForCausalLM,
        tokenizer: LlamaTokenizer,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.device = self.model.device

    def generate(self, x: Dict, max_new_tokens: int = 100) -> List[str]:
        output = self.model.generate(
            inputs=x["input_ids"].to(self.device),
            attention_mask=x["attention_mask"].to(self.device),
            use_cache=True,
            do_sample=True,
            temperature=0.01,
            top_k=50,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prefix = x["input_ids"].shape[1]
        actions = [
            self.tokenizer.decode(output[i, prefix:], skip_special_tokens=True).strip()
            for i in range(x["input_ids"].shape[0])
        ]
        return actions

    def batch_inputs(self, xs: List[Dict]) -> Dict:
        batch = {
            "input_ids": left_pad(
                [x["input_ids"] for x in xs],
                self.tokenizer.pad_token_id,
            ).to("cpu"),
            "attention_mask": left_pad(
                [x["attention_mask"] for x in xs],
                0,
            ).to("cpu"),
        }
        if "labels" in xs[0]:
            batch["labels"] = left_pad(
                [x["labels"] for x in xs],
                IGNORE_INDEX,
            ).to("cpu")
        return batch

    def encode_chat(self, chat: List[Dict]) -> Dict:
        tokens = self.tokenizer.apply_chat_template(
            chat,
            return_tensors="pt",
        )[0]
        return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}

    def encode_chat_with_labels(
        self,
        chat: List[Dict],
    ) -> Dict:
        messages = list(chat)

        tokens = self.tokenizer.apply_chat_template(
            chat,
            return_tensors="pt",
        )[0]
        return {
            "input_ids": tokens,
            "labels": tokens,
            "attention_mask": torch.ones_like(tokens),
        }

        chat_as_string = self.tokenizer.apply_chat_template(messages, tokenize=False)

        instruct_pattern = r"(\[INST\][\s\S]*?\[\/INST\])"

        chat_part = re.split(instruct_pattern, chat_as_string)
        input_ids = []
        labels = []
        for part in chat_part:
            if "[INST]" in part:
                is_instruction = True
            else:
                is_instruction = False
            subpart = part
            if not subpart:
                continue
            if is_instruction:
                part_ids = self.tokenizer(subpart, add_special_tokens=False).input_ids
                input_ids.extend(part_ids)
                labels.extend([IGNORE_INDEX] * len(part_ids))
            else:
                part_ids = self.tokenizer(subpart, add_special_tokens=False).input_ids
                input_ids.extend(part_ids)
                labels.extend(part_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }

    def eval(self):
        self.model.eval()
        self.model.config.use_cache = True

    def train(self):
        self.model.train()
        self.model.config.use_cache = False

    @classmethod
    def load_from_path(
        cls,
        model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.1",
        model_max_length: int = 4096,
    ) -> "MistralChatAgent":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=model_max_length,
            use_fast=False,
        )
        fix_tokenizer(tokenizer)

        model = load_lora_model(MistralForCausalLM, model_name_or_path)
        agent = cls(model, tokenizer)
        return agent
