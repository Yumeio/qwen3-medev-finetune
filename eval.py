import os
import torch
import evaluate
import sacrebleu

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from config import get_config

def evaluate_model(strategy="lora"):
    model_args, data_args, training_args, lora_args, qlora_args, ia3_args, grpo_args, gen_args = get_config(strategy=strategy)

    print(f"Loading model: {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left" 
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=compute_dtype,
        device_map=model_args.device_map,
    )

    # Load Adapter if exists
    adapter_path = training_args.output_dir
    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        print(f"Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()

    print(f"Loading validation data from {data_args.validation_file}...")
    extension = data_args.validation_file.split(".")[-1]
    if extension == "jsonl": extension = "json"
    dataset = load_dataset(extension, data_files={"validation": data_args.validation_file})["validation"]

    if data_args.max_eval_samples:
        dataset = dataset.select(range(data_args.max_eval_samples))

    print("Generating predictions...")
    predictions = []
    references = []
    sources = []

    for batch in tqdm(dataset):
        if "messages" in batch:
            messages = batch["messages"]
            source_content = ""
            reference_content = ""
            for msg in messages:
                if msg["role"] == "user":
                    source_content = msg["content"]
                elif msg["role"] == "assistant":
                    reference_content = msg["content"]
            
            input_messages = [msg for msg in messages if msg["role"] != "assistant"]
            prompt = tokenizer.apply_chat_template(input_messages, tokenize=False, add_generation_prompt=True)
            
        elif "prompt" in batch and "response" in batch:
            source_content = batch["prompt"]
            reference_content = batch["response"]
            prompt = f"<|im_start|>user\n{source_content}<|im_end|>\n<|im_start|>assistant\n"
        else:
            continue

        sources.append(source_content)
        references.append(reference_content)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_args.max_new_tokens,
                do_sample=False, # Use greedy for evaluation usually, or consistent sampling
                temperature=0.0, # Greedy
                top_p=1.0,
            )
        
        # Decode only the new tokens
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predictions.append(prediction)

    # 5. Calculate BLEU Score
    print("Calculating BLEU Score...")
    bleu = evaluate.load("sacrebleu")
    results_bleu = bleu.compute(predictions=predictions, references=references)
    print(f"BLEU Score: {results_bleu['score']:.2f}")



    # Save Results
    with open(os.path.join(training_args.output_dir, "eval_results.txt"), "w", encoding="utf-8") as f:
        f.write(f"BLEU Score: {results_bleu['score']:.2f}\n")
        f.write("\n\n--- Sample Predictions ---\n")
        for i in range(min(5, len(predictions))):
            f.write(f"Source: {sources[i]}\n")
            f.write(f"Ref: {references[i]}\n")
            f.write(f"Pred: {predictions[i]}\n")
            f.write("-" * 20 + "\n")
            
    print(f"Evaluation complete. Results saved to {os.path.join(training_args.output_dir, 'eval_results.txt')}")

if __name__ == "__main__":
    import sys
    strategy = sys.argv[1] if len(sys.argv) > 1 else "lora"
    evaluate_model(strategy=strategy)
