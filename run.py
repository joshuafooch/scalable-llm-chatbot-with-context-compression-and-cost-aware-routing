import gradio as gr
from router import route_query, CONFIDENCE_THRESHOLD
from prompt import build_prompt, add_recent_message, conversation_state

with gr.Blocks() as demo:
    gr.Markdown("# Cost-Aware Query Routing LLM Chatbot")
    gr.Markdown(
        f"""This chatbot uses a small model for simple queries and a larger, 
        more expensive model for complex ones. If the small model's confidence 
        falls below a threshold ({CONFIDENCE_THRESHOLD}), it automatically 
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
        user_message_content = history[-1]["content"][0]["text"] # Get last user input str
        add_recent_message("user", user_message_content, conversation_state)
        prompt = build_prompt(conversation_state)
        print(prompt)
        
        final_response_text, model_used, confidence_str, total_latency, total_cost = route_query(prompt)
        
        metadata = (
            f"**Model Used:** {model_used}\n\n"
            f"**Confidence Score:** {confidence_str}\n\n"
            f"**Latency:** {total_latency:.2f} seconds\n\n"
            f"**Estimated Cost:** ${total_cost:.6f}"
        )

        add_recent_message("assistant", final_response_text, conversation_state)
        history.append({"role": "assistant", "content": final_response_text})
        return history, metadata

    msg.submit(user_message, [msg, chatbot], [msg, chatbot]).then(
        get_bot_response, chatbot, [chatbot, metadata_display]
    )

if __name__ == "__main__":
    demo.launch()