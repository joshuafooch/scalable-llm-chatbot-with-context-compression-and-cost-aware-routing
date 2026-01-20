import json
from config import SYSTEM_GOAL

OUTPUT_FORMAT_INSTRUCTION = """
Provide a response and a corresponding confidence
score indicating how confident you are that the
answer meets the intial query. If unable to provide
a concrete answer, give a low score.
Respond in JSON format:
```json
{
"response": "...",
"confidence": float
}
```
"""

conversation_state = {
    "system_goal": SYSTEM_GOAL,
    "summary": {},
    "recent_messages": [],
    "total_tokens": 0
}

def build_prompt(state: dict) -> list[dict]:
    prompt = []

    prompt.append({
        "role": "system",
        "content": state["system_goal"]
    })

    prompt.append({
        "role": "system",
        "content": f"Conversation summary:\n{json.dumps(state['summary'])}"
    })

    prompt.extend(state["recent_messages"])
    prompt.append({
        "role": "system",
        "content": OUTPUT_FORMAT_INSTRUCTION
    })
    return prompt

def add_recent_message(user: str, message: str, state: dict) -> None:
    state["recent_messages"].append({
        "role": user,
        "content": message
    })

def extract_json_response(response):
    return json.loads(response.split("```json")[-1].split("```")[0])