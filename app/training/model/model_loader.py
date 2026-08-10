from __future__ import annotations

import logging

from transformers import LlamaForCausalLM
from transformers import LlamaConfig

from app.config.schema import ModelPreset, ModelsConfig

logger = logging.getLogger("app.train.model_loader")

# =====================================================================
# Special token id — khớp app/tokenizer/vocab_builder.py
# (SPECIAL_TOKENS_IN_ID_ORDER = [<unk>, <bos>, <eos>, <pad>] -> id 0,1,2,3)
# =====================================================================
PAD_TOKEN_ID = 3
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2

class ModelLoader:
    def __init__(self, cfg: ModelsConfig, preset_name: str) -> None:
        self.cfg = cfg
        self.preset = cfg.presets[preset_name]
        
    def _init_from_scratch(self) -> LlamaForCausalLM:
        llamaConfig = LlamaConfig(
            vocab_size=self.cfg.vocab_size,
            hidden_size=self.preset.hidden_size,
            intermediate_size=self.preset.intermediate_size,
            num_hidden_layers=self.preset.num_hidden_layers,
            num_attention_heads=self.preset.num_attention_heads,
            num_key_value_heads=self.preset.num_key_value_heads,
            max_position_embeddings=self.cfg.max_position_embeddings,
            pad_token_id=PAD_TOKEN_ID,
            bos_token_id=BOS_TOKEN_ID,
            eos_token_id=EOS_TOKEN_ID,
            tie_word_embeddings=True,
        )

        return LlamaForCausalLM._from_config(llamaConfig, attn_implementation="sdpa")
    
    def _load_model_with_vocab_check(self, source: str):
        model = LlamaForCausalLM.from_pretrained(source)
        if model.config.vocab_size != self.cfg.vocab_size:
            raise ValueError(
                f"vocab_size của checkpoint tại {source!r} ({model.config.vocab_size}) KHÔNG khớp "
                f"vocab_size tokenizer hiện tại ({self.cfg.vocab_size}) — vocab đã đổi từ lần train checkpoint "
                f"này, không thể dùng an toàn (vi phạm vocab contract, xem docs/tokenizer_v0.1.md mục 3)."
            )
        return model
    
    def build_model(
        self,
        resume_checkpoint: str = None,
    ) -> LlamaForCausalLM:
        if resume_checkpoint is not None:
            logger.info(f"Resume training from checkpoint: {resume_checkpoint}")
            return self._load_model_with_vocab_check(resume_checkpoint)

        return self._init_from_scratch()
    
    def build_continue_model(
        self,
        resume_checkpoint: str,
        source_repo: str,
    ) -> LlamaForCausalLM:
        if resume_checkpoint is not None:
            logger.info(f"Resume training from checkpoint: {resume_checkpoint}")
            return self._load_model_with_vocab_check(resume_checkpoint)
        
        from huggingface_hub import repo_exists
        if not repo_exists(source_repo):
            raise RuntimeError(
                f"Chưa có checkpoint nào để resume, VÀ source_repo {source_repo!r} chưa tồn tại "
                f"trên Hub — phase này cần checkpoint từ source_repo đã train xong làm nguồn init."
            )
        
        logger.info(f"Chưa có checkpoint nào — bắt đầu từ source_repo: {source_repo}")
        return self._load_model_with_vocab_check(source_repo)
    
    def load_ref_model(
        self,
        source_repo: str,
    ) -> LlamaForCausalLM:
        return self._load_model_with_vocab_check(source_repo)