import json
import random

def get_data(data_path,domain_index):

    with open(data_path, 'r') as f:
        graph = json.load(f)
    users = list(graph.keys())

    domain_num = len(graph[users[0]])
    domain_indexes = range(domain_num)
    domain_users = []
    for domain_index in domain_indexes:
        user_ids = [ user_id for user_id in graph.keys() if graph[user_id][domain_index] ]
        domain_users.append(user_ids)
    
    return graph, domain_users

class DataIterator:
    def __init__(
        self,
        data_path,
        batch_size,
        max_len,
        domain_num,
        domain_index,
        is_train=True,
        only_test_last_one=False,
    ):

        self.batch_size = batch_size
        self.is_train = is_train
        self.max_len = max_len
        self.domain_num = domain_num
        self.domain_index = domain_index
        self.is_train = is_train
        self.only_test_last_one = only_test_last_one

        self.graph, self.domain_users = get_data(data_path, self.domain_index)
        self.users = self.domain_users[self.domain_index]
        self.user_num = len(self.users)
        self.test_index = 0

    
    def __iter__(self):
        return self
    
    def __next__(self):

        if self.is_train:
            user_ids = random.sample(self.users, k=self.batch_size)
        else:
            if self.test_index >= self.user_num:
                self.test_index = 0
                raise StopIteration
            user_ids =  self.users[self.test_index : self.test_index+self.batch_size]
            self.test_index += self.batch_size
        
        item_ids = []
        domain_labels = []
        history_item_ids = []
        history_item_mask = []
        history_max_len = 0

        for user_id in user_ids:

            items = self.graph[user_id]
            single_domain_items = items[self.domain_index]
            item_num = len(single_domain_items)

            if item_num == 0:
                raise ValueError('can not find any history items')
            
            domain_labels.append(self.domain_index)
            mask_value = 1

            if self.is_train:
                if item_num>1:
                    split = random.choice(range(1,item_num))
                    item_ids.append(single_domain_items[split])
                else:
                    split = 0
                    item_ids.append(0)
            else:
                if self.only_test_last_one:
                    split = item_num-1
                else:
                    split = int(item_num * 0.8)
                item_ids.append(single_domain_items[split:])

            if split < self.max_len:
                history_item_ids.append(single_domain_items[:split] + [0] * (self.max_len-split))
                history_item_mask.append([mask_value] * split +  [0] * (self.max_len-split))
                history_max_len = max(history_max_len, split)
            else:
                history_item_ids.append(single_domain_items[split-self.max_len :split])
                history_item_mask.append([mask_value] * self.max_len)
                history_max_len = max(history_max_len, self.max_len)
                
            assert len(history_item_ids) == len(history_item_mask)
        
        # fixed_len_target_ids = [target_ids[:target_max_len] for target_ids in target_item_ids]
        user_ids_int = [int(user_id) for user_id in user_ids]

        return (user_ids_int, item_ids, domain_labels), (history_item_ids, history_item_mask)