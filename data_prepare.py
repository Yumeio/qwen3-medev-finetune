import os
import json
import random
import re
import unicodedata
import argparse
from datasets import load_dataset
from typing import List, Dict, Literal
from tqdm import tqdm
from project_utils import set_seed
import pandas as pd

OUTPUT_RAW_DIR = "dataset/raw"
OUTPUT_DIR = "dataset/processed"

SYSTEM_PROMPT = (
    "Bạn là Qwen-Med, một trợ lý AI chuyên về y học. "
    "Nhiệm vụ của bạn là hỗ trợ dịch thuật chính xác và giải thích các thuật ngữ y khoa "
    "giữa tiếng Anh và tiếng Việt. Hãy luôn trả lời sai cho chuyên nghiệp, ngắn gọn, chính xác và dễ hiểu nhất."
)

PROMPT_ENG_TO_VIE = [
    "Dịch câu văn y học sau sang tiếng Việt:",
    "Chuyển ngữ nội dung này sang tiếng Việt:",
    "Hãy dịch thuật ngữ chuyên ngành này sang tiếng Việt:",
    "Phiên dịch đoạn văn y học sau sang tiếng Việt:",
    "Nghĩa tiếng Việt của đoạn văn này là gì:",
    "Dịch sang Tiếng Việt đoạn văn sau:",
    
    "Translate the following medical text to Vietnamese:",
    "Convert this content into Vietnamese:",
    "Please translate this medical term into Vietnamese:",
]

PROMPT_VIE_TO_ENG = [
    "Dịch câu văn y học sau sang tiếng Anh:",
    "Chuyển ngữ nội dung này sang tiếng Anh:",
    "Hãy dịch thuật ngữ chuyên ngành này sang tiếng Anh:",
    "Phiên dịch đoạn văn y học sau sang tiếng Anh:",
    "Nghĩa tiếng Anh của đoạn văn này là gì:",
    "Dịch sang Tiếng Anh đoạn văn sau:",
    
    "Translate the following medical text to English:",
    "Convert this content into English:",
    "Please translate this medical term into English:",
]

set_seed()

def clean_text(text: str) -> str:

    # Chuẩn hóa Unicode
    text = unicodedata.normalize("NFKC", text)
    
    # Loại bỏ ký tự không mong muốn và chuẩn hóa khoảng trắng
    text = re.sub(r'[\u200b\u200e\u200f\ufeff]', '', text)
    
    # Thay thế mọi loại khoảng trắng (tab, xuống dòng, space lạ) bằng một space thường
    text = re.sub(r'[\s\t\n\r\xa0]+', ' ', text)
    
    # Loại bỏ các ký tự đặc biệt thừa ở đầu và cuối câu
    text = re.sub(r'^[\-\*\•\>]\s+', '', text)
    text = re.sub(r'\s+[\-\*\•\>]$', '', text)
    
    # Chuẩn hóa dấu câu
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    
    # Thêm khoảng trắng sau dấu câu nếu thiếu
    text = re.sub(r'([,.!?;:])(?=[a-zA-Z0-9])', r'\1 ', text)
    
    return text.strip()

def format_chatml(
    messages: List[Dict[str, str]]
) -> str:
    chatml_text = ""
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        chatml_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    return chatml_text.strip()
    
def prepare_train_dataset(output_format: Literal["jsonl", "parquet"] = "parquet"):

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_file = os.path.join(
        OUTPUT_DIR, 
        f"train.{output_format}"
    )
    
    print(f"Preparing training dataset (format: {output_format})...")

    ds = load_dataset(
        "text",
        data_files={
            "en": os.path.join(OUTPUT_RAW_DIR, "train.en.txt"),
            "vi": os.path.join(OUTPUT_RAW_DIR, "train.vi.txt"),
        },
        streaming=False
    )

    en_raw = ds["en"]["text"]
    vi_raw = ds["vi"]["text"]
    
    total_initial = len(en_raw)
    print(f"Initial Dataset: {total_initial:,}")

    # --- Step 1: Cleaning & Punctuation Filtering ---
    # We clean first, then check if valid.
    print("Performing Step 1: Cleaning & Punctuation Filtering...")
    cleaned_pairs = []
    
    for en, vi in tqdm(zip(en_raw, vi_raw), total=total_initial, desc="Cleaning"):
        en_clean = clean_text(en)
        vi_clean = clean_text(vi)
        
        # Punctuation Filtering logic: 
        # Check if empty after cleaning or contains no alphanumeric characters
        if not en_clean or not vi_clean:
            continue
            
        # Optional: strictly check for at least one alphanumeric char
        if not re.search(r'[a-zA-Z0-9]', en_clean) or not re.search(r'[a-zA-Z0-9]', vi_clean):
            continue
            
        cleaned_pairs.append((en_clean, vi_clean))
        
    count_after_punct = len(cleaned_pairs)
    removed_punct = total_initial - count_after_punct
    print(f"-> After Punctuation Filtering: {count_after_punct:,} (Removed: {removed_punct:,})")

    # --- Step 2: De-duplication ---
    print("Performing Step 2: De-duplication...")
    seen = set()
    deduplicated_pairs = []
    
    for en, vi in cleaned_pairs:
        pair = (en, vi)
        if pair not in seen:
            seen.add(pair)
            deduplicated_pairs.append(pair)
            
    count_after_dedup = len(deduplicated_pairs)
    removed_dedup = count_after_punct - count_after_dedup
    print(f"-> After De-duplication: {count_after_dedup:,} (Removed: {removed_dedup:,})")

    # --- Step 3: Quality-Based Filtering ---
    print("Performing Step 3: Quality-Based Filtering...")
    final_pairs = []
    
    for en, vi in deduplicated_pairs:
        # 1. Length Check (Word count)
        en_words = en.split()
        vi_words = vi.split()
        
        len_en = len(en_words)
        len_vi = len(vi_words)
        
        # Min length check
        if len_en < 2 or len_vi < 2:
            continue
            
        # Max length check (optional, but good for quality)
        if len_en > 256 or len_vi > 256:
            continue
            
        # 2. Length Ratio Check
        # Avoid pairs where one is significantly longer than the other
        ratio = len_en / len_vi if len_vi > 0 else 0
        if ratio < 0.25 or ratio > 4.0:
            continue
            
        # 3. Language/Characters Check (Basic) - heuristic
        # If English text has too many Vietnamese specific chars or vice versa?
        # For now, relying on length ratio is a strong baseline.
        
        final_pairs.append((en, vi))
        
    count_final = len(final_pairs)
    removed_quality = count_after_dedup - count_final
    print(f"-> After Quality-Based Filtering: {count_final:,} (Removed: {removed_quality:,})")
    
    # --- Summary Table ---
    print("\n" + "="*40)
    print(f"{'Step':<25} | {'Remaining':<10}")
    print("-" * 40)
    print(f"{'Initial Dataset':<25} | {total_initial:<10,}")
    print(f"{'Punctuation Filtering':<25} | {count_after_punct:<10,} (-{removed_punct})")
    print(f"{'De-duplication':<25} | {count_after_dedup:<10,} (-{removed_dedup})")
    print(f"{'Quality-Based Filtering':<25} | {count_final:<10,} (-{removed_quality})")
    print("="*40 + "\n")

    # --- Convert to ChatML ---
    print("Formatting to ChatML...")
    chatml_data = []
    
    for en_text, vi_text in tqdm(final_pairs, desc="Docs Formatting"):
        # ENG to VIE
        eng_to_vie_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{random.choice(PROMPT_ENG_TO_VIE)}\n{en_text}"},
            {"role": "assistant", "content": vi_text}
        ]
        eng_to_vie_text = format_chatml(eng_to_vie_messages)

        # VIE to ENG
        vie_to_eng_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{random.choice(PROMPT_VIE_TO_ENG)}\n{vi_text}"},
            {"role": "assistant", "content": en_text}
        ]
        vie_to_eng_text = format_chatml(vie_to_eng_messages)
        
        # Add to list
        chatml_data.append({"text": eng_to_vie_text})
        chatml_data.append({"text": vie_to_eng_text})

    random.shuffle(chatml_data)

    
    if output_format == "jsonl":
        print(f"Saving as JSONL to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in chatml_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    elif output_format == "parquet":
        print(f"Saving as Parquet to {output_file}...")
        df = pd.DataFrame(chatml_data)
        df.to_parquet(output_file, index=False, compression='snappy')
    else:
        raise ValueError(f"Unsupported format: {output_format}")
    
    print(f"\n✅ Training data saved to {output_file}")
    print(f"   Format: {output_format.upper()}")
    print(f"   Total Samples (Augmented): {len(chatml_data):,}")
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"   File size: {file_size:.2f} MB")

def prepare_validation_dataset(output_format: Literal["jsonl", "parquet"] = "parquet"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_file = os.path.join(
        OUTPUT_DIR, 
        f"validation.{output_format}"
    )
    
    print(f"\nPreparing validation dataset (format: {output_format})...")

    ds = load_dataset(
        "text",
        data_files={
            "en": os.path.join(OUTPUT_RAW_DIR, "public_test.en.txt"),
            "vi": os.path.join(OUTPUT_RAW_DIR, "public_test.vi.txt"),
        },
        streaming=False
    )

    total_rows = len(ds["en"]) 
    en_list = ds["en"]["text"]
    vi_list = ds["vi"]["text"]
    
    chatml_data = []
    processed_count = 0
    skipped_count = 0

    print("Processing and formatting to ChatML...")
    for i in tqdm(range(total_rows)):
        try:
            en_text = clean_text(en_list[i])
            vi_text = clean_text(vi_list[i])

            if not en_text or \
                not vi_text or \
                len(en_text) < 2\
                or len(vi_text) < 2:
                skipped_count += 1
                continue

            # ENG to VIE
            eng_to_vie_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{random.choice(PROMPT_ENG_TO_VIE)}\n{en_text}"},
                {"role": "assistant", "content": vi_text}
            ]
            eng_to_vie_text = format_chatml(eng_to_vie_messages)

            # VIE to ENG
            vie_to_eng_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{random.choice(PROMPT_VIE_TO_ENG)}\n{vi_text}"},
                {"role": "assistant", "content": en_text}
            ]
            vie_to_eng_text = format_chatml(vie_to_eng_messages)
            
            # Add to list
            chatml_data.append({"text": eng_to_vie_text})
            chatml_data.append({"text": vie_to_eng_text})
            
            processed_count += 2
            
        except Exception as e:
            print(f"Error processing row {i}: {e}") # Optionally print errors
            skipped_count += 1
            continue

    random.shuffle(chatml_data)

    if output_format == "jsonl":
        print(f"Saving as JSONL to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in chatml_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
    elif output_format == "parquet":
        print(f"Saving as Parquet to {output_file}...")
        df = pd.DataFrame(chatml_data)
        df.to_parquet(output_file, index=False, compression='snappy')
    
    else:
        raise ValueError(f"Unsupported format: {output_format}")

    print(f"\n✅ Validation data saved to {output_file}")
    print(f"   Format: {output_format.upper()}")
    print(f"   Processed: {processed_count:,} samples")
    print(f"   Skipped: {skipped_count:,} samples")
    
    # Show file size
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"   File size: {file_size:.2f} MB")

if __name__ == "__main__":
    prepare_train_dataset()
    prepare_validation_dataset()