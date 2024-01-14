from typing import Dict, List

from transformers import (
    AutoTokenizer,
    PhiForCausalLM,
    CodeGenTokenizerFast,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    StoppingCriteria,
)
import torch

from llm_gym.constants import IGNORE_INDEX
from llm_gym.model_utils import (
    fix_tokenizer,
    load_model,
    load_lora_model,
)
from llm_gym.agents.chat_agent import BaseChatAgent


DEFAULT_MODEL_PATH = "microsoft/phi-2"


class _Phi2ChatStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer: CodeGenTokenizerFast):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        texts = self.tokenizer.batch_decode(input_ids)
        return texts[0].endswith("</assistant")


class Phi2ChatAgent(BaseChatAgent):
    def __init__(
        self,
        model: PhiForCausalLM,
        tokenizer: CodeGenTokenizerFast,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.device = "cuda"
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

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
            stopping_criteria=[_Phi2ChatStoppingCriteria(self.tokenizer)],
        )
        prefix = x["input_ids"].shape[1]
        actions = [
            self.tokenizer.decode(output[i, prefix:], skip_special_tokens=True)
            .replace("</assistant", "")
            .strip()
            for i in range(x["input_ids"].shape[0])
        ]
        return actions

    def batch_inputs(self, features: List) -> Dict:
        return self.data_collator(features)

    def encode_chat(self, chat: List[Dict]) -> Dict:
        enc = self.encode_chat_with_labels(
            chat,
            include_prefix=True,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

    def encode_chat_with_labels(
        self,
        chat: List[Dict],
        include_prefix: bool = False,
    ) -> Dict:
        messages = [
            {"role": "user", "content": "What are you?"},
            {
                "role": "assistant",
                "content": "I am a large language model trained for a variety of tasks.",
            },
        ] + list(chat)

        input_ids = []
        labels = []
        for m in messages:
            if m["role"] == "user":
                text = f"<user>{m['content'].strip()}</user>\n"
                part_ids = self.tokenizer(text, add_special_tokens=False).input_ids
                input_ids.extend(part_ids)
                labels.extend([IGNORE_INDEX] * len(part_ids))
            else:
                text = f"<assistant>{m['content'].strip()}</assistant>\n"
                part_ids = self.tokenizer(text, add_special_tokens=False).input_ids
                input_ids.extend(part_ids)
                labels.extend(part_ids)

        if include_prefix:
            text = "<assistant>"
            part_ids = self.tokenizer(text, add_special_tokens=False).input_ids
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
    ) -> "Phi2ChatAgent":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        fix_tokenizer(tokenizer)

        model = load_model(
            AutoModelForCausalLM,
            model_name_or_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        agent = cls(model, tokenizer)
        return agent


class Phi2LoRAChatAgent(Phi2ChatAgent):
    @classmethod
    def load_from_path(
        cls,
        model_name_or_path: str = DEFAULT_MODEL_PATH,
    ) -> "Phi2LoRAChatAgent":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        fix_tokenizer(tokenizer)

        model = load_lora_model(
            AutoModelForCausalLM,
            model_name_or_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            lora_target_modules=["q_proj", "k_proj", "v_proj"],
        )
        agent = cls(model, tokenizer)
        return agent
