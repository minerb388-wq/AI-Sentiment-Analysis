import gzip
import json

file_path = "data/raw/Video_Games.jsonl.gz"

with gzip.open(file_path, "rt", encoding="utf-8") as file:
    first_line = file.readline()

review = json.loads(first_line)

print("First review:")
print(json.dumps(review, indent=4, ensure_ascii=False))

print("\nAvailable fields:")
for field in review.keys():
    print("-", field)