SYSTEM_GOAL = """You are an in-house legal adviser to the X & Y Legal Firm.
"""
OUTPUT_FORMAT_INSTRUCTION = """Provide a response and a corresponding confidence
score indicating how confident you are of the answer.
Respond in JSON format:
```json
{
"response": "This is my response",
"confidence": 0.50
}
```
"""

conversation_state = {
    "system_goal": SYSTEM_GOAL,
    "recent_messages": []
}

def build_prompt(state: dict) -> list[dict]:
    prompt = []
    prompt.append({
        "role": "system",
        "content": state["system_goal"]
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
    return response.split("```json")[-1].split("```")[0]