
import json
import random

filenames = [ "veryeasy.jsonl", "easy.jsonl", "hard.jsonl", "medium.jsonl" ]


numskipped = 0

all_samples = []
for fn in filenames:
    with open(fn) as f:
        samples = f.readlines()
        all_samples.extend(samples)

random.shuffle(all_samples)
print("NUM SAMPLES",len(all_samples))

print("DOING TRAIN DATA")
with open("data/train.jsonl","w") as train_file:
    numdone = 0
    while numdone < 300:
        sample = all_samples.pop(0)
        try:
            x = json.loads(sample)  # just to verify that it is valid json
            train_file.write(sample)
        except:
            numskipped += 1
            print("NUMSKIPPED",numskipped, len(all_samples))
            continue
        numdone += 1

print("DOING VALID DATA")
with open("data/valid.jsonl","w") as valid_file:
    numdone = 0
    while numdone < 10:
        sample = all_samples.pop(0)
        try:
            x = json.loads(sample)  # just to verify that it is valid json
            valid_file.write(sample)
        except:
            numskipped += 1
            print("NUMSKIPPED",numskipped, len(all_samples))
            continue
        numdone += 1

print("DOING TEST DATA")
with open("data/test.jsonl","w") as test_file:
    numdone = 0
    while numdone < 10:
        sample = all_samples.pop(0)
        try:
            x = json.loads(sample)  # just to verify that it is valid json
            test_file.write(sample)
        except:
            numskipped += 1
            print("NUMSKIPPED",numskipped, len(all_samples))
            continue
        numdone += 1

print("DONE")
