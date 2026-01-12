from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Model Load (First Time)

model_path = "./EXAONE-4.0-1.2B"  # Modify to suit your user path

model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()
if torch.cuda.is_available():
    model.to("cuda")

# Reset conversation history
chat_history = [
    {"role": "system", "content": "- The name of the AI language model is \"EXAONE\" and it was created by Naver."},
]

print("Welcome to EXAONE. Type 'exit' to exit.\n")

while True:
    user_input = input("user: ")

    if user_input.lower().strip() in ["exit", "quit"]:
        print("Exit.")
        break

    # Add to conversation history
    chat_history.append({"role": "user", "content": user_input})

    # Tokenizing and input processing
    inputs = tokenizer.apply_chat_template(
        chat_history,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=2048,
            do_sample=True,
            top_p=0.4,
            temperature=0.8,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Remove up to the last user message from the response (because it was put in a template)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    ai_response = response.replace(prompt_text, "").strip()

    # Print and save records
    print(f"EXAONE: {ai_response}\n")
    chat_history.append({"role": "assistant", "content": ai_response})
