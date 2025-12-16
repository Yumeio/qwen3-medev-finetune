import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import get_config

def infer():
    # 1. Load Configuration from config.py
    model_args, _, training_args, lora_args, qlora_args, ia3_args, gen_args, _ = get_config()

    print(f"Loading base model: {model_args.model_name_or_path}")
    
    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )
    
    # 3. Load Base Model
    # Determine device and torch_dtype
    torch_dtype = torch.bfloat16 if model_args.torch_dtype == "bfloat16" or training_args.bf16 else torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=model_args.device_map,
    )

    adapter_path = training_args.output_dir
    
    # Check if adapter config exists in output_dir
    import os
    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        print(f"Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print(f"No adapter found at {adapter_path}. Using base model.")

    model.eval()

    # 5. Inference Loop
    print("\n\n*** Inference Mode ***")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower().strip() == "exit":
            break
            
        # Format input (Qwen Chat template)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=gen_args.max_new_tokens,
                do_sample=gen_args.do_sample,
                temperature=gen_args.temperature,
                top_k=gen_args.top_k,
                top_p=gen_args.top_p,
                repetition_penalty=gen_args.repetition_penalty
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"Assistant: {response}")

if __name__ == "__main__":
    infer()
