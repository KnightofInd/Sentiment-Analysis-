def confidence_check_node(state):
    confidence = state["confidence"]
    state["route"] = "high" if confidence >= 0.80 else "low"
    return state