def generate_response(model, tokenizer, prompt, min_length=3, max_length=100, num_variants=5 , device = 'cuda'):
    """
    Generate multiple responses from the model, ensuring a minimum length.

    Args:
    - model: The language model
    - tokenizer: The model's tokenizer
    - prompt: Input prompt
    - min_length: Minimum length of generated response
    - max_length: Maximum length of generated response
    - num_variants: Number of response variants to generate

    Returns:
    - List of responses that meet the minimum length requirement
    """
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    # Generate multiple responses in one call
    outputs = model.generate(
        **inputs,
        min_length=min_length,
        max_length=max_length,
        do_sample=True,
        temperature=1.2,
        top_k=50,
        top_p=0.95,
        num_return_sequences=num_variants,
        no_repeat_ngram_size=2
    )
    
    # Decode and filter responses
    responses = []
    for output in outputs:
        decoded_response = tokenizer.decode(output, skip_special_tokens=True)
        
        # Extract answer if applicable (adjust logic to your use case)
        if "A:" in decoded_response:
            answer = decoded_response.split("A:")[1].split("\n")[0].strip()
        else:
            answer = decoded_response.strip()
        
        if len(answer) >= min_length:
            responses.append(answer)
    
    # If no valid responses, retry with relaxed constraints
    if not responses:
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=1.3,
            top_k=60,
            top_p=0.97,
            num_return_sequences=num_variants,
            no_repeat_ngram_size=2
        )
        
        for output in outputs:
            decoded_response = tokenizer.decode(output, skip_special_tokens=True)
            
            # Extract answer if applicable
            if "A:" in decoded_response:
                answer = decoded_response.split("A:")[1].split("\n")[0].strip()
            else:
                answer = decoded_response.strip()
            
            if len(answer) >= min_length:
                responses.append(answer)
    
    # If still no responses, return a default message
    if not responses:
        return ["I apologize, but I couldn't generate a meaningful response."]
    
    return responses
