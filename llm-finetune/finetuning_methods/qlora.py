import yaml
from llm_finetune.config.qlora import QLoRAConfig
from dataclasses import asdict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)

class QLoRA_Engine:
    def __init__(self, path_to_model_config) -> None:
        self.qlora_config = self._generate_config(path_to_model_config=path_to_model_config)
        self.model = self._generate_model_from_config(config=self.qlora_config)
        self.tokenizer = self._generate_tokenizer(config=self.qlora_config)

    def _generate_config(path_to_model_config: str):
        config_dict = yaml.safe_load(open(path_to_model_config))
        return QLoRAConfig(**config_dict)
    
    def _generate_model_from_config(config: QLoRAConfig):
        return AutoModelForCausalLM.from_pretrained(**asdict(config))
    
    def _generate_tokenizer(config: QLoRAConfig):
        pass
