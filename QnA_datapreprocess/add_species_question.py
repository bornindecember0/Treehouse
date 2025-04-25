import json
from pathlib import Path

with open("flattened_llama_cub_qa.json", "r") as f:
    flat_qa = json.load(f)

with open("label2species.json", "r") as f:
    label_to_species = json.load(f)  # should be dict of str -> species string

new_species_qa = []
seen = set()

for item in flat_qa:
    key = (item["file_name"], item["label"])
    if key in seen:
        continue
    seen.add(key)
    label = str(item["label"])
    if label in label_to_species:
        species_name = label_to_species[label].replace("_", " ")
        new_species_qa.append({
            "file_name": item["file_name"],
            "description": item["description"],
            "label": item["label"],
            "question": "What species is this bird?",
            "answer": species_name
        })

# combine species QA with existing QA
final_qa = flat_qa + new_species_qa

with open("flattened_llama_cub_qa_with_species.json", "w") as f:
    json.dump(final_qa, f, indent=2)

print("Done")
