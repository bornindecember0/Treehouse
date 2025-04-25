import json
import random

input_file = "flattened_llama_cub_qa_with_species.json"
train_out = "train_qa.json"
val_out = "val_qa.json"

with open(input_file, "r") as f:
    data = json.load(f)

random.shuffle(data)

# split 2/3 train, 1/3 val
split_index = int(len(data) * 2 / 3)
train_data = data[:split_index]
val_data = data[split_index:]

with open(train_out, "w") as f:
    json.dump(train_data, f, indent=2)

with open(val_out, "w") as f:
    json.dump(val_data, f, indent=2)

print(f"Train{len(train_data)}, Val{len(val_data)}")
