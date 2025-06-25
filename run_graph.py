from langgraph.graph import StateGraph, END
from typing import TypedDict
import logging
from datetime import datetime
import os

from rich import print
from rich.console import Console
from rich.prompt import Prompt

# Import your custom nodes
from nodes.inference_node import inference_node
from nodes.confidence_check_node import confidence_check_node
from nodes.fallback_node import fallback_node

# Define the expected state structure
class State(TypedDict, total=False):
    text: str
    prediction: str
    confidence: float
    final_label: str
    route: str  # used internally for routing if fallback is needed

# Setup logging
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/sentiment.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Build LangGraph
builder = StateGraph(State)

# Add nodes
builder.add_node("inference", inference_node)
builder.add_node("check_confidence", confidence_check_node)
builder.add_node("fallback", fallback_node)

# Define execution flow
builder.set_entry_point("inference")
builder.add_edge("inference", "check_confidence")
builder.add_conditional_edges(
    "check_confidence",
    lambda state: state["route"],  # route determined by confidence
    {
        "high": END,
        "low": "fallback"
    }
)
builder.add_edge("fallback", END)

# Compile graph
graph = builder.compile()

# Rich Console
console = Console()

# Run CLI loop
if __name__ == "__main__":
    console.print("[bold blue]🤖 Sentiment Analyzer CLI[/bold blue]")
    console.print("Type 'exit' to quit.\n")

    while True:
        text = Prompt.ask("[yellow]📝 Enter a sentence[/yellow]")
        if text.lower() == "exit":
            console.print("[red]Exiting...[/red]")
            break

        result = graph.invoke({"text": text})
        label = result.get("final_label", result["prediction"])
        confidence = result["confidence"]
        fallback_triggered = "final_label" in result

        console.print(f"\n[bold green]✅ Final Sentiment:[/bold green] {label}")
        console.print(f"[cyan]📊 Confidence:[/cyan] {confidence:.2f}\n")

        logging.info(
            f'Text="{text}" | '
            f'Prediction="{result["prediction"]}" | '
            f'Confidence={confidence:.2f} | '
            f'Fallback={"Yes" if fallback_triggered else "No"} | '
            f'Final="{label}"'
        )