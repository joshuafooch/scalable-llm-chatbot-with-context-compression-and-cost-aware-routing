## Scalable LLM Chatbot with Context Compression and Cost-Aware Routing

A chatbot built on routing between two different models (small and large) with the following features:

- Cost-aware model routing based on confidence threshold
- Context compression via rolling message summarization
- Bounded token growth for long conversations
- Retrieval augmented generation (RAG) implemented using vector database
- Guardrails
- Quantitative evaluation of latency and costs (estimates)

### Setup

To run the chatbot demonstration:

1.  **Create a Virtual Environment** (recommended):

    ```bash
    python3 -m venv .chatbot-env
    ```

2.  **Activate the Virtual Environment**:

    - On macOS/Linux:
      ```bash
      source .chatbot-env/bin/activate
      ```
    - On Windows:
      ```bash
      .chatbot-env\Scripts\activate
      ```

3.  **Install Dependencies**:
    Install all required Python packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuration**:
    Open `config.py` to amend the filepath to the RAG **PDF** reference (`DOCUMENT_FILEPATH`). Otherwise, the provided example handbook is used by default.

### Running the Chatbot

After completing the setup, you can launch the Gradio-based chatbot:

```bash
python run.py
```

Open the provided local URL in your web browser to interact with the chatbot.
