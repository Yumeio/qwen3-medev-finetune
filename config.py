import os
from typing import Literal
from transformers import HfArgumentParser
from arguments import (
    ModelArgs,
    DataTrainingArgs,
    TrainingArguments,
    LoraArgs,
    QloraArgs,
    IA3Args,
    GRPOArgs,
    GenerationArgs
)

def get_config(strategy: Literal["lora", "qlora", "ia3", "grpo"] = "lora"):
    """
    Loads configuration from YAML files based on the specified strategy.
    
    Args:
        strategy: The finetuning strategy ('lora', 'qlora', 'ia3', 'grpo').
                  Defaults to 'lora'.
    """
    
    config_file_map = {
        "lora": "configs/lora.yaml",
        "qlora": "configs/qlora.yaml",
        "ia3": "configs/ia3.yaml",
        "grpo": "configs/grpo.yaml"
    }

    config_file = config_file_map.get(strategy)
    
    if not config_file:
        raise ValueError(f"Invalid strategy '{strategy}'. Supported strategies: {list(config_file_map.keys())}")
        
    print(f"Loading configuration from: {config_file}")

    parser = HfArgumentParser((
        ModelArgs, 
        DataTrainingArgs, 
        TrainingArguments, 
        LoraArgs, 
        QloraArgs, 
        IA3Args, 
        GRPOArgs,
        GenerationArgs
    ))

    # Parse the YAML file
    import yaml
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)

    if "training_args" in config_dict:
        if isinstance(config_dict["training_args"], dict):
             # Merge into top level if it contains args (unlikely based on my read, but safer)
             for k, v in config_dict["training_args"].items():
                 if k not in config_dict:
                     config_dict[k] = v
        
        del config_dict["training_args"]

    model_args, data_args, training_args, lora_args, qlora_args, ia3_args, grpo_args, gen_args = parser.parse_dict(config_dict)

    return model_args, data_args, training_args, lora_args, qlora_args, ia3_args, grpo_args, gen_args
