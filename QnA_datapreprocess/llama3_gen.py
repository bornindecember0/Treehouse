import re
import json
import requests
from pathlib import Path
import time
import os

data_path = Path("cub_bird_dataset.json")
output_path = Path("llama_cub_bird_dataset_with_qa.json")
autosave_path = Path("tmp_autosave_qa.json")

if autosave_path.exists():
    print(f"[Auto-Resume] Loading autosaved file: {autosave_path}")
    with open(autosave_path, "r") as f:
        data = json.load(f)
else:
    with open(data_path, "r") as f:
        data = json.load(f)

def is_multi_answer_question(question):
    keywords = ["color", "colour", "wing", "beak", "head", "body", "tail", "breast", "back", "eye"]
    return any(k in question.lower() for k in keywords)

# clean
def clean_answer(answer, question):
    answer = answer.strip()
    answer = re.sub(r'\(.*?\)', '', answer).strip()
    answer = re.split(r'[-–]', answer)[0].strip()
    answer = answer.replace(" and ", "|").replace(" or ", "|").replace("&", "|")

    raw_parts = [p.strip() for p in re.split(r'[|,]', answer) if p.strip()]
    cleaned = []
    for part in raw_parts:
        tokens = part.strip().strip('"').split()
        if is_multi_answer_question(question):
            cleaned.append(" ".join(tokens))
        else:
            cleaned.append(tokens[0])
    cleaned = list(dict.fromkeys([p for p in cleaned if p]))

    if len(cleaned) == 1:
        return cleaned[0]
    elif is_multi_answer_question(question):
        return cleaned
    else:
        return cleaned[0]


def extract_json_array(output):
    try:
        json_str = re.search(r'\[.*\]', output, re.DOTALL).group(0)
        qa_pairs = json.loads(json_str)
        for qa in qa_pairs:
            qa["answer"] = clean_answer(qa["answer"], qa["question"])
        return qa_pairs
    except Exception as e:
        print("[Parse error]", e)
        print("[Raw output]", output)
        return []

# USE LOCAL LLAMA3
def generate_qa_ollama(description, model="llama3"):
    prompt = f"""
You are a helpful assistant for bird visual question answering.

Given the bird description below, generate 3-5 diverse QA pairs.
Ask about its color, size, body parts, behaviors, and overall appearance.

ONLY respond with a JSON list like this:
[
  {{"question": "...", "answer": "..."}},
  ...
]

Do not use "or", "/", "and", or "&" or " to list multiple answers. Provide a single best answer or separate answers clearly.

Bird description:
\"{description}\"
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate", # MY LOCAL LLAMA3
            json={"model": model, "prompt": prompt, "stream": False}
        )
        output = response.json()["response"]
        return extract_json_array(output)
    except Exception as e:
        print("[failed]", e)
        return []

# autosave
for idx, item in enumerate(data):
    if "qa_pairs" in item and item["qa_pairs"]:
        continue  # skip 
    print(f"[{idx+1}/{len(data)}] Generating QA for {item['file_name']}")
    description = item.get("description", "")
    if description.strip() == "":
        item["qa_pairs"] = []
        continue

    qa_pairs = generate_qa_ollama(description)
    item["qa_pairs"] = qa_pairs

    if idx % 1000 == 0:
        with open(autosave_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[AutoSave] saved progress to {autosave_path}")

    time.sleep(0.9)

with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"QA generation complete AND Saved to: {output_path}")
