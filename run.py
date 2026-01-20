import gradio as gr
from router import route_query, CONFIDENCE_THRESHOLD
from prompt import build_prompt, add_recent_message, conversation_state
from summarizer import should_summarize, summarize_convo
from vectordb import initialize_and_load_vector_db
from config import DOCUMENT_FILEPATH, COLLECTION_NAME, TOP_K_RETRIEVAL

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
        total_latency, total_cost = 0, 0

        # Get last user input string and update to conversation state
        user_message_content = history[-1]["content"][0]["text"]
        add_recent_message("user", user_message_content, conversation_state)

        # Retrieve relevant information from last query (RAG)
        collection, embeddings_model = initialize_and_load_vector_db(DOCUMENT_FILEPATH, COLLECTION_NAME)
        query_embedding = embeddings_model.embed_query(user_message_content)
        retrieved = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K_RETRIEVAL
        )

        # Check length of conversation state to trigger summarization
        if should_summarize(conversation_state):
            summary, summary_latency, summary_cost = summarize_convo(conversation_state)
            conversation_state["summary"] = summary
            total_latency += summary_latency
            total_cost += summary_cost
            print("Summarized!")

        prompt = build_prompt(conversation_state, retrieved["documents"][0])
        
        final_response_text, model_used, confidence_str, latency, cost, total_tokens = route_query(prompt)
        total_latency += latency
        total_cost += cost
        conversation_state["total_tokens"] = total_tokens

        metadata = (
            f"**Model Used:** {model_used}\n\n"
            f"**Confidence Score:** {confidence_str}\n\n"
            f"**Latency:** {total_latency:.2f} seconds\n\n"
            f"**Estimated Cost:** ${total_cost:.6f}"
        )

        add_recent_message("assistant", final_response_text, conversation_state)
        history.append({"role": "assistant", "content": final_response_text})
        print(conversation_state)
        return history, metadata

    msg.submit(user_message, [msg, chatbot], [msg, chatbot]).then(
        get_bot_response, chatbot, [chatbot, metadata_display]
    )

if __name__ == "__main__":
    demo.launch()