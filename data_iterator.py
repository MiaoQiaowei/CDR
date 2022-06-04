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
        user_ids = [ user_id for user_id in graph.keys() if graph[user_id][domain_index] ]
        domain_users.append(user_ids)
    
    return graph, domain_users, domain_indexes

# def get_data(source):
#     self_graph = {}
#     self_users = set()
#     import json
#     with open(source, "r") as f:
#         self_graph = json.load(f)
#     self_users = list(self_graph.keys())
#     self_domain_users = []
#     self_max_domain_num = len(self_graph[self_users[0]])
#     for domain_idx in range(self_max_domain_num):
#         domain_users = [x for x in self_graph.keys() if self_graph[x][domain_idx]]
#         assert len(domain_users) > 0, "at least one user has history in domain {}".format(domain_idx)
#         self_domain_users.append(domain_users)
#     return self_graph, self_domain_users, self_max_domain_num


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
        self.users = self.users[self.domain_index]
        self.user_num = len(self.users)
        self.test_index = 0

    
    def __iter__(self):
        return self
    
    def __next__(self):
        '''
        对数据划分，一个user的前80%进行训练，后20%进行测试
        返回
        '''

        if self.is_train:
            user_ids = random.sample(self.users, k=self.batch_size)
        else:
            if self.test_index >= self.user_num:
                self.test_index = 0
                raise StopIteration
            user_ids =  self.users[self.test_index : self.test_index+self.batch_size]
            self.test_index += self.batch_size

        # if self.train_flag == 0:
        #     # user_id_list = random.sample(self.users, self.batch_size)
        #     user_id_list = []
        #     for domain_idx in range(self.domain_num):
        #         domain_idx = domain_idx if self.domain_idx == 0 else self.domain_idx - 1
        #         user_id_list.extend(random.sample(self.domain_users[domain_idx], self.batch_size // self.domain_num + 1))
        #     user_id_list = user_id_list[:self.batch_size]
        #     assert len(user_id_list) == self.batch_size, "user id list length must be equal to batch size."
        # else:
        #     total_user = len(self.users)
        #     if self.index >= total_user:
        #         self.index = 0
        #         raise StopIteration
        #     user_id_list = self.users[self.index: self.index+self.eval_batch_size]
        #     self.index += self.eval_batch_size
        
        # 对具体的数据进行处理
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
                split = random.choice(range(1,item_num))
                item_ids.append(single_domain_items[split])
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




if __name__ =='__main__':
    # get_data('C:\\CODEs\\CrossDomainRec\\CDR\\data\\ml_nf\\train.json')
    val_loader = DataIterator(
            data_path='data/ml_nf/test.json',
            batch_size=128,
            max_len=20,
            domain_num=1,
            domain_index=0,
            is_train=False,
            use_vqvae=False
        )
    
    for x,y in val_loader:
        # print(x[0])
        # print(x[1])
        # if len(x[1]) == 0:
        #     exit()
        print(len(x[0]))
        print(len(x[1]))
        print(len(x[2]))
        # print(x[1])
            
    