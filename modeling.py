from typing import List
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import LoraConfig, get_peft_model

_DECODER_ONLY = {"gpt2", "opt", "llama", "mistral", "falcon", "qwen", "gemma"}


def _is_decoder_only(model_type: str) -> bool:
    return model_type.lower() in _DECODER_ONLY


def _auto_lora_targets(model_type: str) -> List[str]:
    mt = model_type.lower()
    if mt in {"bert", "roberta", "albert", "deberta", "deberta-v2", "xlm-roberta"}:
        return ["query", "key", "value", "dense"]
    if mt in {"llama", "mistral", "qwen", "gemma", "falcon"}:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if mt in {"gpt2", "opt"}:
        return ["c_attn", "q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "c_fc", "c_proj"]
    return ["query", "key", "value", "dense", "q_proj", "k_proj", "v_proj", "o_proj"]


def load_model_and_tokenizer(
    model_name: str = "facebook/opt-125m",
    num_labels: int = 2,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target: str = "auto",
):
    cfg = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if _is_decoder_only(getattr(cfg, "model_type", "")):
        tok.padding_side = "left"
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
            cfg.pad_token_id = tok.eos_token_id
    else:
        tok.padding_side = "right"
        if tok.pad_token_id is None and getattr(tok, "sep_token_id", None) is not None:
            tok.pad_token = tok.sep_token
            cfg.pad_token_id = tok.pad_token_id

    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)
    if getattr(model.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id

    if use_lora:
        if lora_target.strip() == "" or lora_target.lower() == "auto":
            target_modules = _auto_lora_targets(getattr(cfg, "model_type", ""))
        else:
            target_modules = [t.strip() for t in lora_target.split(",") if t.strip()]
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_cfg)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    return model, tok
