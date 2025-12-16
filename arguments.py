import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union
from transformers import TrainingArguments as HfTrainingArguments

@dataclass
class ModelArgs:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen3-1.7B",
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    cache_dir: Optional[str] = field(
        default="./model_cache",
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust the remote code when loading the model."}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Enables using huggingface_hub token from environment variables."}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}
    )
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={"help": "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the dtype will be automatically derived from the model's weights."}
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device map for model loading. 'auto' will dispatch layers across available devices."}
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "The attention implementation to use (e.g., 'flash_attention_2')."}
    )

@dataclass
class DataTrainingArgs:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default="./dataset/processed/train.parquet",
        metadata={"help": "The input training data file (a jsonl, csv or text file)."}
    )
    validation_file: Optional[str] = field(
        default="./dataset/processed/validation.parquet",
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a jsonl, csv or text file)."}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=16,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."}
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Whether to use packing for SFTTrainer."}
    )

@dataclass
class LoraArgs:
    """
    Arguments for LoRA (Low-Rank Adaptation) configuration.
    """
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA."}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA attention dimension."}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha."}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout."}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "List of module names or regex expression of the module names to replace with LoRA."}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'."}
    )
    lora_task_type: str = field(
        default="CAUSAL_LM",
        metadata={"help": "Task type for LoRA."}
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."}
    )

@dataclass
class QloraArgs:
    """
    Arguments for QLoRA (Quantized LoRA) configuration.
    """
    use_qlora: bool = field(
        default=False,
        metadata={"help": "Whether to use QLoRA."}
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the model in 4-bit precision."}
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the model in 8-bit precision."}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type (fp4 or nf4)."}
    )
    bnb_4bit_use_double_quant: bool = field(
        default=True,
        metadata={"help": "Whether to use double quantization."}
    )
    bnb_4bit_compute_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4-bit base model."}
    )

@dataclass
class IA3Args:
    """
    Arguments for IA3 (Instruction-Adaptive Attention) configuration.
    """
    use_ia3: bool = field(
        default=False,
        metadata={"help": "Whether to use IA3."}
    )
    ia3_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "List of module names or regex expression of the module names to replace with IA3."}
    )
    ia3_feedforward_modules: List[str] = field(
        default_factory=lambda: ["gate_proj", "up_proj", "down_proj"],
        metadata={"help": "List of module names to be treated as feedforward modules for IA3."}
    )
    ia3_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of modules to be set as trainable and saved in the final checkpoint."}
    )

@dataclass
class GRPOArgs:
    """
    Arguments for GRPO (Generative Rejection Policy Optimization).
    """
    use_grpo: bool = field(
        default=False,
        metadata={"help": "Whether to use GRPO."}
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter for the GRPO loss."}
    )
    num_generations: int = field(
        default=4,
        metadata={"help": "Number of generations per prompt."}
    )
    max_prompt_length: int = field(
        default=512,
        metadata={"help": "Maximum length of the prompt."}
    )
    max_completion_length: int = field(
        default=512,
        metadata={"help": "Maximum length of the completion."}
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "The clip parameter for the GRPO loss."}
    )

@dataclass
class GenerationArgs:
    """
    Arguments for generation/inference.
    """
    max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum number of new tokens to generate."}
    )
    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether to use sampling; use greedy decoding otherwise."}
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "The value used to module the next token probabilities."}
    )
    top_k: int = field(
        default=50,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k-filtering."}
    )
    top_p: float = field(
        default=0.9,
        metadata={"help": "If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation."}
    )
    repetition_penalty: float = field(
        default=1.1,
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."}
    )

@dataclass
class TrainingArguments(HfTrainingArguments):
    """
    Custom TrainingArguments to set some defaults or add new ones if needed.
    """
    output_dir: str = field(
        default="./outputs",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    optim: str = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."}
    )
    per_device_train_batch_size: int = field(
        default=128,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={"help": "The initial learning rate for AdamW."}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit."}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture."}
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={"help": "Max gradient norm."}
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use."}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X updates steps."}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy."}
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Save checkpoint every X updates steps."}
    )
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "The evaluation strategy."}
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Run evaluation every X updates steps."}
    )
    dataloader_num_workers: int = field(
        default=16,
        metadata={"help": "Number of subprocesses to use for data loading (PyTorch only)."}
    )
    report_to: Optional[List[str]] = field(
        default_factory=lambda: ["wandb"],
        metadata={"help": "The list of integrations to report the results and logs to."}
    )