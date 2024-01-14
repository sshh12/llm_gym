from typing import Dict, List
import re

from transformers import (
    AutoTokenizer,
    MistralForCausalLM,
    PreTrainedTokenizer,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
)
import torch

from llm_gym.constants import IGNORE_INDEX
from llm_gym.model_utils import (
    fix_tokenizer,
    load_model,
    load_lora_model,
    load_qlora_model,
)
from llm_gym.agents.chat_agent import BaseChatAgent


DEFAULT_MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.1"


class MistralChatAgent(BaseChatAgent):
    def __init__(
        self,
        model: MistralForCausalLM,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.device = "cuda"
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

    def generate(self, x: Dict, max_new_tokens: int = 100) -> List[str]:
        import time

        start = time.time()
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
        elapsed = time.time() - start
        print(output[0, prefix:].shape[0] / elapsed)
        return actions

    def batch_inputs(self, features: List) -> Dict:
        return self.data_collator(features)

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

    @classmethod
    def load_from_path(
        cls,
        model_name_or_path: str = DEFAULT_MODEL_PATH,
    ) -> "MistralChatAgent":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
        )
        fix_tokenizer(tokenizer)

        model = load_model(AutoModelForCausalLM, model_name_or_path)
        agent = cls(model, tokenizer)
        return agent


class MistralLoRAChatAgent(MistralChatAgent):
    @classmethod
    def load_from_path(
        cls,
        model_name_or_path: str = DEFAULT_MODEL_PATH,
    ) -> "MistralChatAgent":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
        )
        fix_tokenizer(tokenizer)

        model = load_lora_model(AutoModelForCausalLM, model_name_or_path)
        agent = cls(model, tokenizer)
        return agent


class MistralQLoRAChatAgent(MistralChatAgent):
    @classmethod
    def load_from_path(
        cls,
        model_name_or_path: str = DEFAULT_MODEL_PATH,
    ) -> "MistralChatAgent":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
        )
        fix_tokenizer(tokenizer)

        model = load_qlora_model(
            AutoModelForCausalLM,
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        agent = cls(model, tokenizer)
        return agent
