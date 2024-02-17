from dataclasses import dataclass, field
from typing import Union, Optional, Any
import torch


@dataclass
class BitsAndBytesConfig:
    load_in_4bit: bool
    load_in_8bit: bool = True
    llm_int8_threshold: float
    llm_int8_has_fp16_weight: bool = False
    bnb_4bit_compute_dtype: Any
    bnb_4bit_use_double_quant: bool
    bnb_4bit_quant_type: str

@dataclass
class QLoRAConfig:
    path_to_model: str
    cache_dir: str
    load_in_4bit: bool
    load_in_8bit: bool = True,
    device_map: str = "auto",
    max_memory: dict
    quantization_config: BitsAndBytesConfig
    torch_dtype: Any
    trust_remote_code: Optional[bool]
    use_auth_token: Optional[bool]
