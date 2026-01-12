from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version
from peft import PeftModel
from packaging import version
import torch
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================
# Paths / knobs
# ==========================
BASE_MODEL_PATH = "./EXAONE-4.0-1.2B"
LORA_PATH       = "./EXAONE4-history-lora"   # output of training script
DOCS_PATH       = "nh_raw.jsonl"             # output of 01_crawl_nh_raw.py

ENABLE_THINKING = False   # True => reasoning mode prompt (opens <think> block), see model card.
MAX_NEW_TOKENS  = 2048


def _check_transformers_version():
    if version.parse(transformers_version) < version.parse("4.54.0"):
        raise RuntimeError(
            f"Transformers >= 4.54.0 is required for EXAONE 4.0, but you have {transformers_version}. "
            "Please upgrade: pip install -U 'transformers>=4.54.0'"
        )

def _pick_dtype():
    if torch.cuda.is_available():
        try:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


# ==========================
# Document loading & search preparation
# ==========================
def load_docs(path=DOCS_PATH):
    docs = []
    texts = []

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"The file {path} does not exist. Please crawl it first.")

    with p.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            text = d.get("text", "").strip()
            if not text:
                continue
            docs.append(d)
            texts.append(text)

    print(f"[RAG] Loaded {len(docs)} docs from {path}")

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2)
    )
    tfidf = vectorizer.fit_transform(texts)
    print("[RAG] Built TF-IDF matrix:", tfidf.shape)

    return docs, vectorizer, tfidf


def search_docs(query, docs, vectorizer, tfidf_matrix, k=3):
    """
    Returns the top k documents by TF-IDF cosine similarity for a user query.
    """
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix)[0]
    best_idx = scores.argsort()[::-1][:k]
    return [docs[i] for i in best_idx], [scores[i] for i in best_idx]


# ==========================
# Model loading (Base + LoRA)
# ==========================
def load_model_and_tokenizer():
    _check_transformers_version()

    print("[Model] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    # EXAONE provides PAD/BOS/EOS; keep a safety guard anyway.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = _pick_dtype()
    print(f"[Model] Loading base model with dtype={dtype} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=dtype,
        device_map="auto",
    )

    print("[Model] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    return model, tokenizer


def _clean_exaone_text(txt: str) -> str:
    # Remove chat markers if they ever appear in the generated tail.
    for marker in ("[|assistant|]", "[|user|]", "[|system|]", "[|tool|]", "[|endofturn|]"):
        txt = txt.replace(marker, "")
    return txt.strip()


# ==========================
# Main Chat Loop
# ==========================
def main():
    # 1) RAG preparation
    docs, vectorizer, tfidf_matrix = load_docs(DOCS_PATH)

    # 2) Model/Tokenizer Loading
    model, tokenizer = load_model_and_tokenizer()

    # 3) Conversation history (EXAONE chat template supports roles: system/user/assistant/tool)
    chat_history = [
        {
            "role": "system",
            "content": (
                "- You are EXAONE (LG AI Research) fine-tuned for Korean history Q&A.\n"
                "- When answering historical questions, rely primarily on the provided reference texts.\n"
                "- If the answer is not in the references, say that you do not know rather than guessing."
            ),
        },
    ]

    print("Welcome to EXAONE 4.0 (History LoRA + RAG). Type 'exit' to exit.\n")

    while True:
        try:
            user_input = input("User: ")
        except EOFError:
            print("\nExit.")
            break

        if user_input.lower().strip() in ["exit", "quit"]:
            print("Exit.")
            break

        # --------------------------
        # 1) Retrieve related text in our history net documents
        # --------------------------
        retrieved_docs, scores = search_docs(
            user_input, docs, vectorizer, tfidf_matrix, k=3
        )

        context_chunks = []
        for d, s in zip(retrieved_docs, scores):
            snippet = d["text"][:5000]
            title = d.get("title", "")
            url = d.get("url", "")
            context_chunks.append(
                f"[Source Title] {title}\n[URL] {url}\n\n{snippet}"
            )

        context_text = "\n\n------------------------------\n\n".join(context_chunks)

        # --------------------------
        # 2) Build the user message (RAG-augmented)
        # --------------------------
        augmented_user_input = (
            "다음은 국사편찬위원회 우리역사넷(신편 한국사)에서 가져온 참고 자료이다.\n"
            "아래 참고 자료와 사용자 질문을 바탕으로, 우리역사넷의 서술과 모순되지 않도록 신중하게 한국어로 답하라.\n"
            "참고 자료에 없는 내용은 추측하지 말고, 모른다고 대답해도 된다.\n\n"
            f"=== 참고 자료 시작 ===\n{context_text}\n=== 참고 자료 끝 ===\n\n"
            f"질문: {user_input}"
        )

        # Add to history
        chat_history.append({"role": "user", "content": augmented_user_input})

        # --------------------------
        # 3) Tokenize & Generate using EXAONE chat template
        # --------------------------
        inputs = tokenizer.apply_chat_template(
            chat_history,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=ENABLE_THINKING,
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.8,
                top_p=0.4,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Keep only newly generated part
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_len:]
        ai_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        ai_response = _clean_exaone_text(ai_response)

        print(f"EXAONE: {ai_response}\n")

        # Add assistant response to history
        chat_history.append({"role": "assistant", "content": ai_response})


if __name__ == "__main__":
    main()
