
import time
import torch

def generate_question(model, tokenizer, prompt, max_length=100, device="cuda"):
    """Generate a question using Flan-T5 with a dynamic seed based on system time."""
    # Set a seed based on the current system time
    seed = int(time.time())
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate a question
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.9
    )

    # Decode and return the generated question
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question.strip()
