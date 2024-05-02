import random
import json

import torch
import numpy

import dataset
import util
from client import Client
from server import Server

if __name__ == '__main__':

    # 解析命令行参数
    # parser = argparse.ArgumentParser(description='IC_fed_weight')
    # parser.add_argument('-c', '--conf', dest='conf')
    # args = parser.parse_args()

    # 读取配置文件
    with open('conf.json', 'r') as f:
        conf = json.load(f)

    # ----------------------------------------------------------

    # 'fedavg' 'fedavg_EWM' 'fedavg_EWM_DBSCAN'
    mode = conf['mode']
    # 'all_data' 'order_split' 'random_split' 'class_split'
    split_mode = conf['split_mode']
    # 'resnet18' 'resnet34' 'resnet50'
    model_name = conf['model_name']
    # 'cifar10' 'cifar100'
    dataset_name = conf['dataset_name']

    batch_size = conf['batch_size']     # 批次大小
    lr = conf['lr']                     # 学习率
    momentum = conf['momentum']         # 动量
    lambda_eps = conf['lambda_eps']         # 聚类算法邻域参数即epsilon，TODO：这个值的设置很重要
    lambda_mps = conf['lambda_mps']         # 聚类算法最小样本数即min_points，TODO：这个值的设置很重要

    global_epoch_num = conf['global_epoch_num']         # 全局训练次数
    local_epoch_num = conf['local_epoch_num']           # 局部训练次数
    client_num = conf['client_num']                     # 客户端数量
    client_train_ratio = conf['client_train_ratio']                   # 客户端训练验证比例
    '''这里的参数每个类别的客户端最大数量，因为算法的独特设计，真实类别可能小于3，
    但是因此某个客户端某个类别的数据集为会更大'''
    client_per_class = conf['client_per_class']         # 每个类别的客户端数量

    is_set_seed = conf['is_set_seed']                   # 是否设置随机种子
    seed = conf['seed']                                 # 随机种子

    # ----------------------------------------------------------

    print('---------------------------------')
    print('#Start# ')

    # 信息展示
    print(f'Mode: {mode}, ' +
          f'Split mode: {split_mode}, ' +
          f'Model: {model_name}, ' +
          f'Dataset: {dataset_name}. ')

    if torch.cuda.is_available():
        print(f'Using GPU: {torch.cuda.get_device_name(0)}. ')
    else:
        print('Using CPU. ')

    # ----------------------------------------------------------
    
    print('---------------------------------')
    print('#Start process# ')

    # 数据处理

    # 随机种子
    if is_set_seed:
        # random的种子
        random.seed(seed)

    # 目录处理
    util.dir_process()

    # 获取数据集
    train_set, val_set = dataset.get_dataset(dataset_name)

    # 得到标签列表
    label_list = train_set.classes
    print(f'Dataset: {dataset_name}, ' +
          f'Labels: {set(label_list)}. ')
    class_num = len(label_list)

    # 定义设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # ----------------------------------------------------------

    # class_num是总的类别数量，class_list是本地的在类别分类过程中产生的指定类别
    dataset_list = []
    class_list = []

    # 全部模式
    if split_mode == 'all_data':

        for _ in range(client_num):
            dataset_list.append(train_set)

        for _ in range(client_num):
            class_list.append(label_list)
        
    # 顺序独立模式
    elif split_mode == 'order_split':

        data_per_client = int(len(train_set) / client_num)

        for idx in range(client_num):
            indice = list(range(idx * data_per_client, (idx + 1) * data_per_client))
            dataset_list.append(torch.utils.data.Subset(train_set, indice))

        for _ in range(client_num):
            class_list.append(label_list)

    # 随机独立模式
    elif split_mode == 'random_split':

        data_per_client = int(len(train_set) / client_num)
        split = []
        for _ in range(client_num):
            split.append(data_per_client)
        # 剩余部分
        remain = len(train_set) - data_per_client * client_num
        split.append(remain)
        dataset_list = list(torch.utils.data.random_split(train_set, split))
        # 删除剩余部分
        dataset_list.pop(-1)

        for _ in range(client_num):
            class_list.append(label_list)

    # 类别独立模式
    elif split_mode == 'class_split':

        # 数据类别字典，得到类别对应数据集的字典
        dataset_dict = {}
        for label in label_list:
            dataset_dict[label] = []
        for data, label_idx in train_set:
            dataset_dict[label_list[label_idx]].append((data, label_idx))
        
        # 类别字典，即某个类别对应哪些客户端
        temp_dict = {}
        for label in label_list:
            temp_dict[label] = []
            for _ in range(client_per_class):
                temp_dict[label].append(random.randint(0, client_num - 1))

        # 分割与类别记录
        for _ in range(client_num):
            dataset_list.append([])
            class_list.append([])

        for label in label_list:
            temp_len = len(dataset_dict[label])
            data_per_client = int(temp_len / client_per_class)
            for j in range(client_per_class):
                dataset_list[temp_dict[label][j]].extend(
                    dataset_dict[label][j * data_per_client: (j + 1) * data_per_client])
                class_list[temp_dict[label][j]].append(label)

        for idx, data in enumerate(dataset_list):
            dataset_list[idx] = util.MyDataset(data)

        pass

    # ----------------------------------------------------------

    server = Server(device=device, 
                    model_name=model_name, 
                    dataset=val_set,
                    batch_size=batch_size,
                    class_num=class_num)
    client_list = []
    for client_id in range(client_num):
        client_list.append(Client(client_id=client_id, 
                                  device=device, 
                                  model_name=model_name, 
                                  dataset=dataset_list[client_id], 
                                  train_ratio=client_train_ratio,
                                  batch_size=batch_size,
                                  class_num=class_num,
                                  class_list = class_list[client_id]))
        
    print('#Finish process# ')

    # ----------------------------------------------------------

    print('---------------------------------')
    print('#Start train# ')

    # 计时器
    timer = util.Timer()

    for global_epoch_id in range(global_epoch_num):

        # 得到训练后状态字典列表
        state_dict_list = []
        for client_id, client in enumerate(client_list):
            state_dict_list.append(client.local_train(  local_epoch_num = local_epoch_num, 
                                                        device=device, 
                                                        global_epoch_id=global_epoch_id,
                                                        lr=lr,
                                                        momentum=momentum))
        
        # ----------------------------------------------------------

        if mode == 'fedavg':
            
            # 得到聚合分配后状态字典列表
            update_state_dict_list = server.model_integrate_allocate(   state_dict_list=state_dict_list,
                                                                        final_weight=[1/client_num]*client_num)
            
            # 全局评估
            server.global_eval( device=device, 
                                global_epoch_id=global_epoch_id)
            
            # 权值归还
            for client in client_list:
                client.param_return(update_state_dict_list[client.client_id])
            
        elif mode == "fedavg_EWM":

            # [1][2]代表2号客户端的权值在1号客户端数据集上的效果
            # 第一维是客户端，第二维是权值
            origin_matrix = numpy.zeros((client_num, client_num))

            for client in client_list:
                for client_id in range(client_num):
                    client_acc = client.local_eval( device=device,
                                                    state_dict=state_dict_list[client_id])
                    origin_matrix[client.client_id][client_id] = client_acc

            final_weight = util.MYEWM(origin_matrix)

            update_state_dict_list = server.model_integrate_allocate(   state_dict_list=state_dict_list,
                                                                        final_weight=final_weight)
            
            server.global_eval( device=device, 
                                global_epoch_id=global_epoch_id)
            
            for client in client_list:
                client.param_return(update_state_dict_list[client.client_id])

        elif mode == 'fedavg_EWM_DBSCAN':
            
            # 簇列表
            cluster_list = util.MYDBSCAN(   lambda_eps=lambda_eps,
                                            lambda_mps=lambda_mps,
                                            state_dict_list=state_dict_list)
            
            print(f'Cluster: {cluster_list}. ')
            
            # 遍历簇并进行模型整合与分配
            for cluster in cluster_list:

                if cluster is list:

                    temp_state_dict_list = []
                    for idx in cluster:
                        temp_state_dict_list.append(state_dict_list[idx])

                    origin_matrix = numpy.zeros((len(cluster), len(cluster)))

                    for idx_a, idx_b in enumerate(cluster):
                        client = client_list[idx_b]
                        for idx_c, idx_d in enumerate(cluster):
                            client_acc = client.local_eval( device=device,
                                                            state_dict=state_dict_list[idx_d])
                            origin_matrix[idx_a][idx_c] = client_acc

                    final_weight = util.MYEWM(origin_matrix)

                    temp_list = server.model_integrate_allocate(    state_dict_list=temp_state_dict_list,
                                                                    final_weight=final_weight)
                    
                    for idx, client_id in enumerate(cluster):
                        client_list[client_id].param_return(temp_list[idx])
            
            pass

        # ----------------------------------------------------------

        print(f'Time-take: {timer.cal_interval()}. ')
        print('---------------------------------')

    print('#Finish train# ')
    print('---------------------------------')

    print('#End# ')
    print('---------------------------------')





