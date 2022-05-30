import pandas as pd
import argparse
from tqdm import tqdm
import os
import sys
from multiprocessing import Pool, cpu_count
import pathlib
from sklearn.model_selection import train_test_split
import random

target_len = 3
negtive_number = 10

def applyParallel(dfGrouped, func):
    with Pool(8) as p:
        ret_list = p.map(func, tqdm([group for name, group in dfGrouped], desc="process history"))
    return pd.concat(ret_list)

def get_user_history(df):
    domain = df["domain"].tolist()[0]
    user_history = df["item"].tolist()
    user_history_len = len(user_history)
    user_row = {"user": [df["user"].tolist()[0]], "user_history_{}".format(domain): [",".join(user_history)],
                "number_{}".format(domain): [user_history_len]}
    return pd.DataFrame.from_dict(user_row)

def get_user_vocab(users):
    users = ["pad"] + users
    userid2idx = {x:i for i, x in enumerate(users)}
    useridx2id = users
    print("user number: {}".format(len(users) - 1))
    return {"userid2idx": userid2idx, "useridx2id": useridx2id}

def get_item_vocab(items):
    itemidx2id = ["pad", "domain1", "domain2", "eos"]
    itemid2idx = {"pad": 0, "domain1": 1, "domain2": 2, "eos": 3}
    items = [x for y in items for x in y.split(",") if x]
    from collections import Counter
    items_counter = Counter(items)
    items_counter = sorted(items_counter.items(), key=lambda x:x[1], reverse=True)
    for x, _ in items_counter:
        if x not in itemid2idx:
            itemidx2id.append(x)
            itemid2idx[x] = len(itemidx2id) - 1
    print("item number: {}".format(len(itemidx2id)))
    return {"itemidx2id": itemidx2id, "itemid2idx": itemid2idx}




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="Books,Movies_and_TV", help="domain to preprocess, Books,Electronics,Movies_and_TV....")
    parser.add_argument("--debug", default=0, type=int, help="debug mode = 1")
    parser.add_argument("--min_clicks", default=5, type=int, help="min clicks number requirements")
    parser.add_argument("--output_dir", default="data/amazon_data", help="output dir")
    parser.add_argument("--target_num", default=3, type=int, help="target number")
    parser.add_argument("--outer", default=1, type= int, help="whether to keep all training data, domain1 or domain2 has empty sequence")
    args = parser.parse_args()
    domains = args.domains.split(",")
    assert args.min_clicks > target_len
    print("preprocess domains: {}".format(domains))

    domains2ids = {x:idx+1 for idx, x in enumerate(domains)}
    domains_dict = {}
    data_dir = "../data/amazon_data_all"
    for domain in domains:
        csv_file = "{}/ratings_{}.csv".format(data_dir, domain)
        domain_csv = pd.read_csv(csv_file, encoding="utf-8", names=["user", "item", "rating", "time"])
        if args.debug > 0:
            print("debug mode use top 100w samples")
            domain_csv = domain_csv.iloc[:1000000]
        domain_csv["domain"] = domains2ids[domain]
        domain_csv = domain_csv.sort_values(["user", "time"])
        print("start to process history")

        item_clicked_cnt = domain_csv.groupby("item").count().reset_index()
        item_clicked_cnt["i_number"] = item_clicked_cnt["user"]
        item_clicked_cnt = item_clicked_cnt[["item", "i_number"]]
        domain_csv = domain_csv.merge(item_clicked_cnt, on="item", how="left")
        print("before, domain: clicked > {} logs number: {}".format(args.min_clicks, len(domain_csv)))

        domain_csv = domain_csv[domain_csv["i_number"] > 20]
        print("after item filter. domain: clicked > {} logs number: {}".format(args.min_clicks, len(domain_csv)))

        user_clicked_cnt = domain_csv.groupby("user").count().reset_index()
        user_clicked_cnt["u_number"] = user_clicked_cnt["item"]
        user_clicked_cnt = user_clicked_cnt[["user", "u_number"]]
        domain_csv = domain_csv.merge(user_clicked_cnt, on="user", how="left")

        domain_csv = domain_csv[domain_csv["u_number"] > args.min_clicks]

        print("after user filter. domain: clicked > {} logs number: {}".format(args.min_clicks, len(domain_csv)))
        domain_csv = applyParallel(domain_csv.groupby("user"), get_user_history)
        print("user number: {}".format(len(domain_csv)))
        domains_dict[domain] = domain_csv
    # merged_data = domains_dict["Books"].merge(domains_dict["Movies_and_TV"], on="user", how="left")
    if args.outer == 0:
        merged_data = pd.merge(*list(domains_dict.values()), on="user", how="left")
        for domain_id in domains2ids.values():
            merged_data = merged_data[merged_data["user_history_{}".format(domain_id)].notna()]
            merged_data = merged_data[merged_data["user_history_{}".format(domain_id)] != ""]
    else:
        merged_data = pd.merge(*list(domains_dict.values()), on="user", how="outer")
        for domain_id in domains2ids.values():
            merged_data["user_history_{}".format(domain_id)] = merged_data["user_history_{}".format(domain_id)].fillna("")

    users = merged_data["user"].tolist()
    items = []
    for domain_id in domains2ids.values():
        items.extend(merged_data["user_history_{}".format(domain_id)].tolist())
        item_vocab_tmp = get_item_vocab(merged_data["user_history_{}".format(domain_id)].tolist())
        print("domain: {}, item number: {}".format(domain_id, len(item_vocab_tmp["itemid2idx"])))

    user_vocab = get_user_vocab(users)
    item_vocab = get_item_vocab(items)

    print("user number: {}, item number: {}".format(len(user_vocab["userid2idx"]), len(item_vocab["itemid2idx"])))
    train, test = train_test_split(merged_data, test_size=0.3, random_state=123)
    valid, test = train_test_split(test, test_size=0.5, random_state=123)
    data = {}
    pd_data = {"train": train, "valid": valid, "test":test}
    for split, split_data in pd_data.items():
        data = {}
        for idx, row in tqdm(split_data.iterrows(), desc="preprocess {}".format(split)):
            user = user_vocab["userid2idx"][row["user"]]
            history = []
            min_history_len = 999999999
            for domain_id in domains2ids.values():
                user_history = row["user_history_{}".format(domain_id)]
                user_history = [item_vocab["itemid2idx"][x] for x in user_history.split(",") if x]
                history.append(user_history)
                if len(user_history) > 0:
                    min_history_len = min(min_history_len, len(user_history))
            try:
                assert min_history_len >= args.min_clicks
            except:
                print("erro")
            data[user] = history
        with open("{}/{}.json".format(data_dir, split), "w") as f:
            import json
            json.dump(data, f, indent=4)



