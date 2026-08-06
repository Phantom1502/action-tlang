from __future__ import annotations

import warnings
from typing import Optional

from transformers import PreTrainedTokenizerFast

def load_tokenizer(
    repo_id: str,
    revision: Optional[str] = None,
) -> PreTrainedTokenizerFast:
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(repo_id, revision=revision)
        return tok
    except Exception as e:
        raise RuntimeError(
            f"Không load được tokenizer từ Hub (repo_id={repo_id!r}): {e}. "
        ) from e