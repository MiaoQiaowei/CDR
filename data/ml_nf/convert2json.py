import json
from tqdm import tqdm
from collections import Counter

ml_test_file = "./ml.test.rating"
ml_train_file = "./ml.train.rating"
nf_test_file = "./nf.test.rating"
nf_train_file = "./nf.train.rating"

def read_txt_file(file):
    data = {}
    with open(file, "r") as f:
        for line in tqdm(f):
            if "\t" in line:
                line = line.strip("\n").split("\t")
                assert len(line) == 4
            elif "," in line:
                line = line.strip("\n").split(",")
                assert len(line) == 2
            userid = int(line[0])
            itemid = int(line[1])
            data[userid] = data.get(userid, []) + [itemid]
    return data

def join_domains_data(domain1, domain2):
    data = {}
    for userid, history in domain1.items():
        data[userid] = [history, domain2.get(userid, [])]
    for userid, history in domain2.items():
        if userid not in data:
            data[userid] = [domain1.get(userid, []), history]
    return data

def join_train_test_data(train, test):
    for userid, history in test.items():
        test[userid] = [x + y for x, y in zip(train[userid], test[userid])]
    return test

def write_data(data, file):
    with open(file, "w") as f:
        json.dump(data, f, indent=4)

def convert2new_itemid(data, itemsidx2newidx):
    for userid, histories in tqdm(data.items(), total=len(data), desc="conver to new idx"):
        new_histories = []
        for history in histories:
            tmp = [itemsidx2newidx[x] for x in history]
            new_histories.append(tmp[:])
        data[userid] = new_histories[:]
    return data

ml_train_data = read_txt_file(ml_train_file)
nf_train_data = read_txt_file(nf_train_file)

train_data = join_domains_data(ml_train_data, nf_train_data)

ml_test_data = read_txt_file(ml_test_file)
nf_test_data = read_txt_file(nf_test_file)

test_data = join_domains_data(ml_test_data, nf_test_data)
test_data = join_train_test_data(train_data, test_data)

items = []
for userid, histories in test_data.items():
    for history in histories:
        items.extend(history)
items_conter = Counter(items)
items_conter = sorted(items_conter.items(), key=lambda x: x[1], reverse=True)
items = ["pad", "domain_ml", "domain_nf", "end"] + [x[0] for x in items_conter]
print("item number: {}".format(len(items)))
itemsidx2newidx = {itemid:idx for idx, itemid in enumerate(items)}
train_data = convert2new_itemid(train_data, itemsidx2newidx)
test_data = convert2new_itemid(test_data, itemsidx2newidx)

write_data(data=train_data, file="./train.json")
write_data(data=test_data, file="./test.json")
write_data(data=test_data, file="./valid.json") # tmp data

print("done!")