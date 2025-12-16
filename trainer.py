import os
import torch
from typing import Literal
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    IA3Config,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
from config import get_config
from trl import SFTTrainer

try:
    from trl import GRPOTrainer
except ImportError:
    GRPOTrainer = None
    print("Warning: GRPOTrainer not found in trl. Please upgrade trl to the latest version.")

def train(strategy: Literal["qlora", "lora", "ia3", "grpo"] = "lora"):
    model_args, data_args, training_args, lora_args, qlora_args, ia3_args, grpo_args, gen_args = get_config(strategy=strategy)
    
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Output Dir: {training_args.output_dir}")

    # 2. Load Tokenizer
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="right" 
    )
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

    data_files = {}
    if data_args.train_file:
        data_files["train"] = data_args.train_file
    if data_args.validation_file:
        data_files["validation"] = data_args.validation_file

    extension = data_args.train_file.split(".")[-1]
    if extension == "jsonl":
        extension = "json"
    
    dataset = load_dataset(extension, data_files=data_files)

    bnb_config = None
    if qlora_args.use_qlora:
        compute_dtype = getattr(torch, qlora_args.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=qlora_args.load_in_4bit,
            bnb_4bit_quant_type=qlora_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=qlora_args.bnb_4bit_use_double_quant,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        device_map=model_args.device_map,
    )

    if qlora_args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    peft_config = None
    if ia3_args.use_ia3:
        print("Strategy: IA3")
        peft_config = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=ia3_args.ia3_target_modules,
            feedforward_modules=ia3_args.ia3_feedforward_modules,
            modules_to_save=ia3_args.ia3_modules_to_save,
        )
    elif lora_args.use_lora or qlora_args.use_qlora:
        print(f"Strategy: {'QLoRA' if qlora_args.use_qlora else 'LoRA'}")
        peft_config = LoraConfig(
            task_type=lora_args.lora_task_type,
            inference_mode=False,
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            target_modules=lora_args.lora_target_modules,
            bias=lora_args.lora_bias,
            modules_to_save=lora_args.modules_to_save,
        )
    
    if grpo_args.use_grpo:
        print("Strategy: GRPO")
        if GRPOTrainer is None:
            raise ImportError("GRPOTrainer is not available in the installed version of trl.")
            
        print("Starting GRPO training...")
        
        def formatting_prompts_func(example):
            return example["prompt"] if "prompt" in example else example["text"]

        from trl.rewards import accuracy_reward
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            peft_config=peft_config,
            reward_func=accuracy_reward, 
        )
        
    else:
        print("Starting SFT training...")
        
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            peft_config=peft_config,
            tokenizer=qwen_tokenizer,
        )

    request = trainer.train()
    print(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    print("Training complete.")

    print("Starting evaluation...")
    from eval import evaluate_model
    evaluate_model(strategy=strategy)

if __name__ == "__main__":
    import sys
    strategy = "lora"
    train(strategy=strategy)
