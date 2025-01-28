from utils.detection import detect_ai_text

def select_best_response(detection_model, detection_tokenizer, responses):
    """Select the most human-like response based on confidence from AI text detection"""
    best_response = None
    lowest_confidence = 2

    for response in responses:
        detection_result = detect_ai_text(detection_model, detection_tokenizer, response)
        if detection_result['confidence'] < lowest_confidence:
            lowest_confidence = detection_result['confidence']
            best_response = response

    return best_response, lowest_confidence


def select_worst_response(detection_model, detection_tokenizer, responses):
    """Select the most human-like response based on confidence from AI text detection"""
    best_response = None
    most_confidence = -1

    for response in responses:
        detection_result = detect_ai_text(detection_model, detection_tokenizer, response)
        if detection_result['confidence'] > most_confidence:
            most_confidence = detection_result['confidence']
            best_response = response

    return best_response, most_confidence