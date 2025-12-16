# src/split.py
import os, random, json
p = os.path.expanduser("/home/minhchau/anaconda3/envs/curvas/curvas/Preprocessed")
cases = sorted([d for d in os.listdir(p) if os.path.isdir(os.path.join(p,d))])
random.seed(42)
random.shuffle(cases)
n = len(cases)
n_train = int(n * 0.8)
train = cases[:n_train]
val = cases[n_train:]
out = {"train": train, "val": val}
with open("split.json","w") as f:
    json.dump(out,f,indent=2)
print("Saved split.json with",len(train),"train",len(val),"val")
