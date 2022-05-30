
data_dir = "data/amazon_data_all"
train_path = "{}/{}".format(data_dir, "train.json")
test_path = "{}/{}".format(data_dir, "test.json")

import json
with open(train_path, "r") as f:
    train_data = json.load(f)
with open(test_path, "r") as f:
    test_data = json.load(f)

import pdb

d1n = 0
d1_iset = set()
d1_in = 0
d2n = 0
d2_iset = set()
d2_in = 0
for uid, seqs in train_data.items():
    if seqs[0] != []: d1n += 1
    if seqs[1] != []: d2n += 1
    for e in seqs[0]:
        d1_iset.add(e)
    d1_in += len(seqs[0])
    for e in seqs[1]:
        d2_iset.add(e)
    d2_in += len(seqs[1])


pdb.set_trace()










