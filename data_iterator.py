import json
import random
import numpy as np

def get_data(data_path):
    '''
    reture all domain info
    '''
    with open(data_path, 'r') as f:
        graph = json.load(f)
    users = list(graph.keys())

    domain_num = len(graph[users[0]])
    domain_indexes = range(domain_num)
    domain_users = []
    for domain_index in domain_indexes:
        user_ids = [ user_id for user_id in users if graph[user_id][domain_index] ]
        domain_users.append(user_ids)
    
    return graph, domain_users, domain_indexes


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
        use_vqvae=False
    ):

        self.batch_size = batch_size
        self.is_train = is_train
        self.max_len = max_len
        self.domain_num = domain_num
        self.domain_index = domain_index
        self.is_train = is_train
        self.only_test_last_one = only_test_last_one
        self.use_vqvae = use_vqvae

        self.graph, self.users, self.domain_indexes = get_data(data_path)

        self.test_index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        '''
        对数据划分，一个user的前80%进行训练，后20%进行测试
        返回
        '''

        if self.is_train:
            user_ids = random.sample(self.users[self.domain_index], k=self.batch_size)
        else:
            user_ids =  self.users[self.test_index : self.test_index+self.batch_size]
            self.test_index += self.batch_size
        
        # 对具体的数据进行处理
        item_ids = []
        domain_labels = []
        target_item_ids = []
        target_item_mask = []
        target_max_len = 0

        for user_id in user_ids:
            items = self.graph[user_id]
            single_domain_items = items[self.domain_index]
            item_num = len(single_domain_items)

            if item_num == 0:
                raise ValueError('can not find any history items')
            
            domain_labels.append(self.domain_index)

            if self.is_train:
                if self.only_test_last_one:
                    split = item_num-1
                else:
                    split = int(item_num * 0.8)
                item_ids.append(single_domain_items[:split])
            else:
                split = random.choice(range(1,item_num))
                item_ids.append(random.choice(single_domain_items[split]))
            
            if split < self.max_len: 
                target_item_ids.append(single_domain_items[:split] + [0] * self.max_len-split)
                target_item_mask.append(domain_labels * split +  [0] * self.max_len-split)
                target_max_len = max(target_max_len, split)
            else:
                target_item_ids.append(single_domain_items[split-self.max_len :split])
                target_item_mask.append(domain_labels * self.max_len)
                target_max_len = max(target_max_len, self.max_len)
        
        fixed_len_target_ids = [target_ids[:target_max_len] for target_ids in target_item_ids]

        return (user_ids, item_ids, domain_labels, fixed_len_target_ids), (target_item_ids, target_item_mask)




if __name__ =='__main__':
    get_data('C:\\CODEs\\CrossDomainRec\\CDR\\data\\ml_nf\\train.json')
    