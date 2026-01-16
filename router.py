import time
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from prompt import extract_json_response

# --- Load model onto correct device ---
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device
device = get_device()
print("Loading models... This may take a moment.")
small_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
small_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
small_model.to(device)
large_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
large_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
large_model.to(device)
print(f"Models loaded successfully to {device}.")

# --- Cost and Confidence Configuration ---
# Example costs per 1000 tokens for input and output
SMALL_MODEL_COST = {"input": 0.0005, "output": 0.0015}
LARGE_MODEL_COST = {"input": 0.0020, "output": 0.0050}

CONFIDENCE_THRESHOLD = 0.80

def model_handler(model_str: str, messages: list[dict]) -> tuple[str, float, float, float]:
    """
    Handles a call to the LLM model.
    """
    if model_str == "small":
        model = small_model
        tokenizer = small_tokenizer
    elif model_str == "large":
        model = large_model
        tokenizer = large_tokenizer
    
    start_time = time.time()    
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    response = extract_json_response(response)
    print(response)
    response = json.loads(response)

    # Calculate token usage for cost
    input_tokens = inputs["input_ids"].shape[1]
    output_tokens = outputs.shape[1]
    
    cost = (input_tokens / 1000 * SMALL_MODEL_COST["input"]) + \
           (output_tokens / 1000 * SMALL_MODEL_COST["output"])
           
    end_time = time.time()
    latency = end_time - start_time
    
    return response["response"], response["confidence"], latency, cost

def route_query(messages: list[dict]) -> tuple[str, str, str, float, float]:
    """
    Routes the user's query to the appropriate model based on confidence.
    Returns the final response text and metadata components.
    """
    # 1. Always call the small model first
    small_response, confidence, small_latency, small_cost = model_handler("small", messages)
    
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
        large_response, _, large_latency, large_cost = model_handler("large", messages)
        model_used = "Large Model"
        final_response = large_response
        total_latency = small_latency + large_latency # Total time includes both calls
        total_cost = small_cost + large_cost # Total cost includes both calls
        confidence_str = f"{confidence:.2f} (rerouted)"

    return final_response, model_used, confidence_str, total_latency, total_cost
