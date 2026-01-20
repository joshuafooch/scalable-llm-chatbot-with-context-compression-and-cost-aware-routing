import json
from router import summary_model_handler
from prompt import extract_json_response
from config import MAX_TURNS, MAX_TOKENS_TO_SUMMARIZE

SUMMARY_PROMPT = """Summarize the conversation based on the
overarching goal of the user ("user_goal"), the key points of
discussion, any decisions that the user should make, any constraints,
and any open questions ("open_questions") that the user has asked but
have not beed answered. Respond in JSON format:
```json
{
"user_goal": "...",
"key_points": ["..."],
"decisions": ["..."],
"constraints": ["..."],
"open_questions": ["..."]
}
```

Rules:
- Do NOT include chit-chat
- Do NOT speculate
- Preserve technical precision
- Be concise and to the point
- Include key points from every conversation turn
"""

def should_summarize(state: dict) -> bool:
    return (
        len(state["recent_messages"]) >= MAX_TURNS or
        state["total_tokens"] >= MAX_TOKENS_TO_SUMMARIZE
    )

def summarize_convo(state: dict) -> tuple[str, float, float]:
    # Generate summary
    prompt = []
    prompt.append({
        "role": "system",
        "content": f"Conversation summary:\n{json.dumps(state['summary'])}"
    })
    prompt.extend(state["recent_messages"][:-1])  # most recent user input not included
    prompt.append({"role": "system", "content": SUMMARY_PROMPT})
    response, latency, cost = summary_model_handler(prompt)
    response = extract_json_response(response)

    # Remove all previous conversation turns except most recent user input
    state["recent_messages"] = state["recent_messages"][-1:]

    return response, latency, cost