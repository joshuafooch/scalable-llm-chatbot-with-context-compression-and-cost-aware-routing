import gradio as gr
import time
import random

# Example costs of small and large models
SMALL_MODEL_COST = {"input": 0.0005, "output": 0.0015}
LARGE_MODEL_COST = {"input": 0.0020, "output": 0.0050}

# Confidence threshold for rerouting to large model
CONFIDENCE_THRESHOLD = 0.80

# Small model handler
def small_model_handler(message: str, history: list[dict]) -> tuple[str, float, float, float]:
    """
    Simulates a call to a small, fast, and cheap language model.
    """
    start_time = time.time()
    
    # Simulate processing time
    latency = random.uniform(0.1, 0.5)
    time.sleep(latency)
    
    # Simulate a response and a confidence score
    confidence = random.uniform(0.5, 1.0)
    response_text = f"Small model response."
    
    # Simulate token usage for cost calculation
    if isinstance(message, str):
        input_tokens = len(message.split())
    elif isinstance(message, list):
        input_tokens = len(message)
    else:
        input_tokens = 0 # Fallback for unknown types
    output_tokens = len(response_text.split())
    
    cost = (input_tokens / 1000 * SMALL_MODEL_COST["input"]) + \
           (output_tokens / 1000 * SMALL_MODEL_COST["output"])
           
    end_time = time.time()
    latency = end_time - start_time
    
    return response_text, confidence, latency, cost

# Large model handler
def large_model_handler(message: str, history: list[dict]) -> tuple[str, float, float, float]:
    """
    Simulates a call to a large, slow, and expensive language model.
    """
    start_time = time.time()
    
    # Simulate longer processing time
    latency = random.uniform(1.0, 3.0)
    time.sleep(latency)
    
    response_text = f"Large model response."

    # Simulate token usage for cost calculation
    if isinstance(message, str):
        input_tokens = len(message.split())
    elif isinstance(message, list):
        input_tokens = len(message)
    else:
        input_tokens = 0 # Fallback for unknown types
    output_tokens = len(response_text.split())
    
    cost = (input_tokens / 1000 * LARGE_MODEL_COST["input"]) + \
           (output_tokens / 1000 * LARGE_MODEL_COST["output"])
           
    end_time = time.time()
    latency = end_time - start_time
    
    return response_text, 1.0, latency, cost # Large model is always confident

def route_query(message: str) -> tuple[str, str, str, float, float]:
    """
    Routes the user's query to the appropriate model based on confidence.
    Returns the final response text and metadata components.
    """
    # 1. Always call the small model first
    small_response, confidence, small_latency, small_cost = small_model_handler(message, None) # No history passed to models
    
    # 2. Check confidence score
    if confidence >= CONFIDENCE_THRESHOLD:
        # Use small model's response
        model_used = "Small Model"
        final_response = small_response
        total_latency = small_latency
        total_cost = small_cost
        confidence_str = f"{confidence:.2f}"
    else:
        # 3. If confidence is low, call the large model
        large_response, _, large_latency, large_cost = large_model_handler(message, None) # No history passed to models
        model_used = "Large Model"
        final_response = large_response
        total_latency = small_latency + large_latency # Total time includes both calls
        total_cost = small_cost + large_cost # Total cost includes both calls
        confidence_str = f"{confidence:.2f} (rerouted)"

    return final_response, model_used, confidence_str, total_latency, total_cost


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Cost-Aware Query Routing LLM Chatbot")
    gr.Markdown(
        """This chatbot uses a small model for simple queries and a larger, 
        more expensive model for complex ones. If the small model's confidence 
        is below a threshold " ({CONFIDENCE_THRESHOLD}), it automatically 
        routes the query to the large model."""
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat History")
            msg = gr.Textbox(label="Your Message", placeholder="Type your message here...")
            clear = gr.ClearButton([msg, chatbot])
        
        with gr.Column(scale=1):
            metadata_display = gr.Markdown("### Log Data", label="Metadata")

    def user_message(message: str, history: list[dict]) -> tuple[str, list[dict]]:
        return "", history + [{"role": "user", "content": message}]

    def get_bot_response(history: list[dict]) -> tuple[list[dict], str]:
        print(history)
        user_message_content = history[-1]["content"] # Get last user input
        
        final_response_text, model_used, confidence_str, total_latency, total_cost = route_query(user_message_content)
        
        metadata = (
            f"**Model Used:** {model_used}\n\n"
            f"**Confidence Score:** {confidence_str}\n\n"
            f"**Latency:** {total_latency:.2f} seconds\n\n"
            f"**Estimated Cost:** ${total_cost:.6f}"
        )

        history.append({"role": "assistant", "content": final_response_text})
        return history, metadata


    msg.submit(user_message, [msg, chatbot], [msg, chatbot]).then(
        get_bot_response, chatbot, [chatbot, metadata_display]
    )

if __name__ == "__main__":
    demo.launch()