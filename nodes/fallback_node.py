def fallback_node(state):
    text = state["text"]
    from rich import print
    from rich.prompt import Confirm

    print(f"\n[bold yellow]⚠️ Low confidence ({state['confidence']:.2f}) for:[/bold yellow] \"{text}\"")
    is_negative = Confirm.ask("🧐 Do you think this is a [bold red]Negative[/bold red] review?")

    if is_negative:
        state["final_label"] = "Negative"
    else:
        state["final_label"] = state["prediction"]

    return state
