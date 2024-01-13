from typing import List, Any

from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
import torch


def load_model(model_cls: Any, *args) -> Any:
    model = model_cls.from_pretrained(*args, device_map="auto")
    return model


def load_lora_model(model_cls: Any, *args) -> Any:
    config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = model_cls.from_pretrained(*args, device_map="auto")
    model = get_peft_model(model, config)

    return model


def load_qlora_model(model_cls: Any, *args) -> Any:
    config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = model_cls.from_pretrained(*args, device_map="auto", load_in_8bit=True)
    model = get_peft_model(model, config)

    return model


def fix_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.mask_token is None:
        tokenizer.mask_token = tokenizer.unk_token
    if tokenizer.cls_token is None:
        tokenizer.cls_token = tokenizer.unk_token
    if tokenizer.sep_token is None:
        tokenizer.sep_token = tokenizer.unk_token


def left_pad(values: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max(len(ids) for ids in values)

    padded_values = torch.stack(
        [F.pad(ids, (max_len - len(ids), 0), "constant", pad_value) for ids in values]
    )

    return padded_values
