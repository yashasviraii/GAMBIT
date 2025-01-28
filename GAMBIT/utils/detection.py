
import torch
import torch.nn.functional as F

def detect_ai_text(detection_model, detection_tokenizer, text , device = 'cuda'):
    """Detect if text is AI-generated"""
    # Preprocess text (assuming a preprocess function exists)
    try:
        from MAGE.deployment import preprocess
        text = preprocess(text)
    except ImportError:
        print("Warning: MAGE preprocess function not found. Using raw text.")

    # Tokenize the text
    inputs = detection_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    # Forward pass to get probabilities
    with torch.no_grad():
        outputs = detection_model(**inputs)

    # Get probabilities and predicted class
    probs = F.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][predicted_class].item()

    # 0 typically means AI-generated, 1 means human-written
    return {
        'is_ai_generated': predicted_class == 0,
        'confidence': confidence
    }

