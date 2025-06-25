from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pre-trained model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Set model to evaluation mode
model.eval()

# Optional: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_sentiment(text):
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    # Move tensors to the correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    labels = ["Negative", "Neutral", "Positive"]
    prediction_idx = probabilities.argmax()
    prediction = labels[prediction_idx]
    confidence = probabilities[prediction_idx]

    return prediction, confidence

if __name__ == "__main__":
    text = "The movie was painfully slow."
    label, conf = predict_sentiment(text)
    print(f"Prediction: {label}, Confidence: {conf:.2f}")
# Example usage