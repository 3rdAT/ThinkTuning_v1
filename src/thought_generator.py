import ipdb
def gen_thought_prompt(prefix_ids, suffix_ids, model, tokenizer):

    prefix_text, suffix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True), tokenizer.decode(suffix_ids, skip_special_tokens=True)

    user_prompt = f"[prefix]\n{prefix_text}\n[suffix]\n{suffix_text}"
    
    messages = [
        {
            "role":"system",
            "content": "Generate a thought one would think in between the [prefix] and [suffix]"
        },
        {
            "role":"user",
            "content":f"{user_prompt}.\nFormat your answer between [thought] and [/thought]"
        },
    ]

    batch = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = tokenizer(batch, return_tensors="pt", add_special_tokens=False)
    prompt_len = len(batch['input_ids'][0])
    batch = {k: v.to("cuda") for k, v in batch.items()}

    outputs = model.generate(**batch, max_new_tokens=4096, do_sample=True, temperature=0.8, use_cache=True, top_p=1.0)

    output_tokens = outputs.sequences[0][prompt_len:]
    output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)

    thought_prompt = output_text.split("[thought]")[1].split("[/thought]")[0].strip()

    thought_ids = tokenizer(thought_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]

    ipdb.set_trace()

    return thought_ids