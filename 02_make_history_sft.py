import json
from pathlib import Path
from tqdm import tqdm

RAW_PATH = "nh_raw.jsonl"
OUT_PATH = "history_sft_train.jsonl"

# When dividing a very long text, use the following character criteria (roughly):
MAX_CHARS = 1000   # Maximum length of one chunk
MIN_CHARS = 100   # Discard chunks that are too short

INSTRUCTION_TEMPLATE = (
    "다음 글은 국사편찬위원회 우리역사넷(신편 한국사)에 실린 한국사 관련 설명이다. "
    "이 글의 내용을 충실히 따르면서, 한국어로 자연스럽게 서술하라."
)

def split_into_chunks(text: str, max_chars: int):
    """
    Simply cut it by line/paragraph and create chunks that fit within max_chars.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    chunks = []
    current = ""

    for ln in lines:
        # If the sentence is too long, don't split it up, just paste it together.
        # (If you want to be more precise, you can split it again at the period.)
        if len(current) + len(ln) + 1 <= max_chars:
            if current:
                current += "\n" + ln
            else:
                current = ln
        else:
            if current:
                chunks.append(current)
            current = ln

    if current:
        chunks.append(current)

    return chunks

def main():
    raw_path = Path(RAW_PATH)
    if not raw_path.exists():
        raise FileNotFoundError(f"{RAW_PATH} is missing. Please run 01_crawl_nh_raw.py first.")

    cnt_in, cnt_out = 0, 0
    with open(raw_path, encoding="utf-8") as f_in, \
         open(OUT_PATH, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in, desc="Building SFT data"):
            cnt_in += 1
            d = json.loads(line)
            text = d.get("text", "").strip()
            if not text:
                continue

            chunks = split_into_chunks(text, MAX_CHARS)
            for ch in chunks:
                if len(ch) < MIN_CHARS:
                    continue

                record = {
                    "instruction": INSTRUCTION_TEMPLATE,
                    "input": ch,
                    "output": ch,  # Restoration/Mimetic SFT
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                cnt_out += 1

    print(f"[DONE] Input documents {cnt_in} -> Generate SFT samples {cnt_out}, file: {OUT_PATH}")

if __name__ == "__main__":
    main()
