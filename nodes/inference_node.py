from load_model import predict_sentiment

def inference_node(state):
    text = state["text"]
    prediction, confidence = predict_sentiment(text)
    state.update({
        "prediction": prediction,
        "confidence": confidence
    })
    return state