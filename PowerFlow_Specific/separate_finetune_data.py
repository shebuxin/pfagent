import json
import random

data_file = 'finetune_data.jsonl'

with open(data_file, 'r') as f:
    data = [json.loads(line) for line in f]

random.shuffle(data)

# Split ratio (e.g., 80% training, 20% validation)
train_ratio = 0.8
split_index = int(len(data) * train_ratio)

train_data = data[:split_index]
validation_data = data[split_index:]

with open('train.jsonl', 'w') as f:
    for entry in train_data:
        f.write(json.dumps(entry) + "\n")

with open('validation.jsonl', 'w') as f:
    for entry in validation_data:
        f.write(json.dumps(entry) + "\n")

print(f"Total entries: {len(data)}")
print(f"Training set: {len(train_data)} entries")
print(f"Validation set: {len(validation_data)} entries")
