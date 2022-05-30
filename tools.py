import os.path as osp


def get_data_info(args):

    data_info = {}
    data_info['path']= osp.join(args.data_path, args.dataset)
    data_info['dataset'] = args.dataset

    if args.dataset == 'ml_nf':
        domain = ['ml','nf']
        data_info['domain'] = domain[args.domain_index]
        data_info['item_count'] = 5569+10
        data_info['max_len'] = 20
        data_info['test_iter'] = 500

    elif args.dataset == 'tc_iqi':
        domain = ['tc', 'iqi']
        data_info['domain'] = domain[args.domain_index]
        data_info['item_count'] = 4871+10
        data_info['max_len'] = 20
        data_info['test_iter'] = 500

    elif args.dataset == 'amazon':
        domain = ['amazon_1', 'amazon_2']
        data_info['domain'] = domain[args.domain_index]
        data_info['item_count'] = 365318+100
        data_info['max_len'] = 20
        data_info['test_iter'] = 2000

    elif args.dataset == 'amazon_all':
        domain = ['amazon_1', 'amazon_2']
        data_info['domain'] = domain[args.domain_index]
        data_info['item_count'] = 217587+100
        data_info['max_len'] = 20
        data_info['test_iter'] = 1000

    else:
        raise ValueError(f'DATA is not supported')

    return data_info


def get_model(args):
    if args.model == 'DNN':
        from model import DNN
        model = DNN(args)
    else:
        raise ValueError(f'can not find {args.model}')

    return model    