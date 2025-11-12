import torch


def predict_model(model, tokenizer, messages, configuration=None):
    """
    Generate a response using the provided Qwen model/tokenizer pair and a chat-style message list.

    Args:
        model: Pre-trained causal LM (e.g., Qwen/Qwen2-1.5B-Instruct).
        tokenizer: Matching tokenizer for the model.
        messages (list[dict]): Chat history with `role` and `content`.
        configuration (dict, optional): Supports `temperature` and `max_token_limit`.
    """
    if configuration is None:
        configuration = {}

    temperature = configuration.get("temperature", 0.1)
    max_token_limit = configuration.get("max_token_limit", 2000)

    if not messages:
        raise ValueError("The `messages` list cannot be empty.")

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    generation_kwargs = {
        "max_new_tokens": max_token_limit,
        "temperature": temperature,
        "do_sample": temperature > 0,
        "pad_token_id": pad_token_id,
    }

    with torch.no_grad():
        generated_ids = model.generate(model_inputs, **generation_kwargs)

    response_ids = generated_ids[0, model_inputs.shape[-1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    return response


def model_evaluation(model_type, model, tokenizer, system_content, question, formatted_options, configuration=None):
    if model_type == "qwen2" or model_type == "qwen3":
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {question}\n\nOptions:\n{formatted_options}"}
        ]
        model_result = predict_model(model, tokenizer, messages, configuration)
    else: 
        raise ValueError(f"Unknown model_type: {model_type}")

    #  print(f"Model result: {model_result}")
    return model_result
