import math
import pickle
import time
import os
import shutil

from sklearn.cluster import DBSCAN
import numpy

# 目录处理
def dir_process():
    # 如果不存在则创建dataset文件夹
    if os.path.exists('dataset') == False:  
        os.mkdir('dataset')
    # 如果存在则删除log文件夹
    if os.path.exists('log') == True:   
        shutil.rmtree('log')
    os.mkdir('log')

# 解析数据集
def _unpickle(filename, dataset_name):
    dict = None
    if dataset_name == 'cifar10':
        with open(filename, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
    elif dataset_name == 'cifar100':
        with open(filename, 'rb') as f:
            dict = pickle.load(f, encoding='latin1')
    return dict

# 读取数据集，得到标签列表
def get_label(dataset_name):
    if dataset_name == 'cifar10':
        dict = _unpickle(filename='dataset/cifar-10-batches-py/batches.meta',
                             dataset_name=dataset_name)
        label_list = dict[b'label_names']
        label_list = [str(label, encoding='utf-8') for label in label_list]
    elif dataset_name == 'cifar100':
        dict = _unpickle(filename='dataset/cifar-100-python/meta',
                             dataset_name=dataset_name)
        label_list = dict['fine_label_names']
    return label_list

class MyDataset:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        image = self.data[index][0]
        target = self.data[index][1]

        return image, target

    def __len__(self):
        return len(self.data)
    
class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.temp_time = self.start_time
        self.time_list = [self.start_time]
    
    def _time_convert(self, seconds):
        minutes = int(seconds / 60)
        seconds = int(seconds) % 60
        return f'{minutes}m, {seconds}s'
    
    def cal_interval(self):
        self.temp_time = time.time()
        self.time_list.append(self.temp_time)
        return self._time_convert(self.temp_time - self.time_list[-2])
    
    def get_time(self):
        return time.ctime()

# ----------------------------------------------------------
# 熵权法（https://zhuanlan.zhihu.com/p/267259810）

# 按方案进行极值化
def _extreme_method(matrix):
    # 得到每行的最大值
    row_max = []
    row_min = []
    for i in range(len(matrix)):
        row_max.append(max(matrix[i]))
        row_min.append(min(matrix[i]))
    # 极值化
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            # 避免除零错误
            if row_max[i] == row_min[i]:
                matrix[i][j] = 0.5
            else:
                matrix[i][j] = (matrix[i][j] - row_min[i]) / (row_max[i] - row_min[i])
    return matrix

'''这里不会零除，出现零除请告知'''
# 计算矩阵的在指定指标中每个方案所占比重
def _medium_process(matrix):
    # 得到每列的和
    col_sum = []
    for i in range(len(matrix[0])):
        sum = 0
        for j in range(len(matrix)):
            sum += matrix[j][i]
        col_sum.append(sum)
    # 更新矩阵
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = matrix[i][j] / col_sum[j]
    return matrix

# 计算各指标信息熵
def _cal_entropy(matrix):
    # 得到每列的和
    col_sum = []
    for i in range(len(matrix[0])):
        sum = 0
        for j in range(len(matrix)):
            sum += matrix[j][i]
        col_sum.append(sum)
    # 得到每列的信息熵
    col_entropy = []
    for i in range(len(matrix[0])):
        entropy = 0
        for j in range(len(matrix)):
            # 避免log(0)错误
            if matrix[j][i] != 0:
                entropy += matrix[j][i] * math.log(matrix[j][i])
        entropy = entropy / math.log(len(matrix))
        col_entropy.append(-entropy)
    return col_entropy

# 计算信息效用
def _cal_score(col_entropy):
    # 得到信息效用
    col_score = []
    for i in range(len(col_entropy)):
        col_score.append(1 - col_entropy[i])
    return col_score

'''这里不会零除，出现零除请告知'''
# 计算权重
def _cal_weight(col_score):
    col_weight = []
    sum = 0
    for i in range(len(col_score)):
        sum += col_score[i]
    for i in range(len(col_score)):
        col_weight.append(col_score[i] / sum)
    return col_weight

# 计算各方案的综合得分
def _cal_final_score(matrix, col_weight):
    final_score = []
    for i in range(len(matrix)):
        score = 0
        for j in range(len(matrix[0])):
            score += matrix[i][j] * col_weight[j]
        final_score.append(score)
    return final_score

'''这里不会零除，出现零除请告知'''
# 计算各方案的权值
def _cal_final_weight(final_score):
    final_weight = []
    sum = 0
    for i in range(len(final_score)):
        sum += final_score[i]
    for i in range(len(final_score)):
        final_weight.append(final_score[i] / sum)
    return final_weight

# 一行是一个方案、一列是一个指标；这里一行是一个客户端、一列是一个权重
# 熵权法
def MYEWM(origin_matrix):
    matrix = _extreme_method(origin_matrix)
    matrix = _medium_process(matrix)
    col_entropy = _cal_entropy(matrix)
    col_score = _cal_score(col_entropy)
    col_weight = _cal_weight(col_score)
    final_score = _cal_final_score(col_weight = col_weight, matrix = origin_matrix)
    final_weight = _cal_final_weight(final_score)
    return final_weight

# ----------------------------------------------------------
# 聚类手动实现参见博客（https://www.cnblogs.com/pinard/p/6208966.html）
# 现使用sklearn的现成函数进行操作

# 状态字典转向量（返回值：python list）
def _state_dict_to_vector(state_dict):
    vector = []
    for name, param in state_dict.items():
        vector.extend(param.view(-1).tolist())
    return vector

# 聚类算法
def MYDBSCAN(lambda_eps, lambda_mps, state_dict_list):
    # 将state_dict_list转换为numpy array
    state_dict_array = []
    for state_dict in state_dict_list:
        state_dict_array.append(_state_dict_to_vector(state_dict))
    state_dict_array = numpy.array(state_dict_array)

    # 聚类
    db = DBSCAN(eps=lambda_eps, min_samples=lambda_mps).fit(state_dict_array)
    labels = db.labels_

    # 聚类结果转换成期望的形式
    cluster_list = []
    for i in range(max(labels)+1):
        cluster_list.append([])
    # 添加多个列表和单个噪声点，表示分组
    for i in range(len(labels)):
        if labels[i] == -1:
            cluster_list.append(i)
        else:
            cluster_list[labels[i]].append(i)
    
    return cluster_list

# ----------------------------------------------------------

# # 计算L2范数（即距离）
# def _cal_L2_norm(vector1, vector2):
#     sum = 0
#     for i in range(len(vector1)):
#         sum += (vector1[i] - vector2[i]) ** 2
#     return math.sqrt(sum)

# # 计算距离矩阵（L2范数矩阵）
# def _cal_distance_matrix(state_dict_list):
#     distance_matrix = []
#     for i in range(len(state_dict_list)):
#         distance_matrix.append([])
#         for j in range(len(state_dict_list)):
#             vector_i = _state_dict_to_vector(state_dict_list[i])
#             vector_j = _state_dict_to_vector(state_dict_list[j])
#             distance_matrix[i].append(_cal_L2_norm(vector_i, vector_j))
#     return distance_matrix

# # 邻域参数 lambda_b（ε 邻域半径），lambda_c（MinPts 领域点个数）
# def DBSCAN(lambda_eps, lambda_mps, state_dict_list):

#     distance_matrix = _cal_distance_matrix(state_dict_list) # 距离矩阵
#     point_list = list(range(len(state_dict_list)))          # 点列表，从0到n代表每个点（D 样本集）

#     # 步骤1，初始化
#     center_point_list = []                  # 核心点列表（Ω 核心对象集合），因为集合没法随机选择一个，这里直接用列表
#     cluster_num = 0                         # 聚类簇数（k）
#     candidate_point_list = []               # 候选点列表（Γ 未访问样本集合）
#     for point in point_list:
#         candidate_point_list.append(point)
#     cluster_list = []                       # 簇列表（C 簇划分）

#     # 步骤2
#     for point_a in point_list:
#         neighbor_list = []          # 邻域子样本集（Nε(Nj)）
#         for point_b in point_list:
#             if distance_matrix[point_a][point_b] <= lambda_eps:
#                 neighbor_list.append(point_b)
#         if len(neighbor_list) >= lambda_mps:
#             center_point_list.append(point_a)

#     # 步骤3
#     # 如果没有中心点则算法结束
#     while len(center_point_list) != 0:
#         # 步骤4
#         center_point = center_point_list[0]         # 核心对象（o）
#         cluster_center_point_list = [center_point]  # 簇核心对象集合（Ωcur）
#         cluster_num += 1
#         cluster = [center_point]                    # 簇样本集合（Ck）
#         candidate_point_list.remove(center_point)

#         # 步骤5
#         # 如果当前簇核心对象队列长度为0，转入步骤3
#         while len(cluster_center_point_list) != 0:
#             # 步骤5续
#             center_point_list = list(set(center_point_list) - set(cluster))

#             # 步骤6
#             cluster_center_point = cluster_center_point_list[0]          # 簇核心对象（o'）
#             cluster_neighbor_list = []                                   # 簇核心对象邻域样本集（Nε(o')）
#             for point in point_list:
#                 if distance_matrix[cluster_center_point][point] <= lambda_eps:
#                     cluster_neighbor_list.append(point)
#             temp_list = list(set(cluster_neighbor_list) & set(candidate_point_list))    # 临时集合（Δ）
#             cluster = list(set(cluster) | set(temp_list))
#             candidate_point_list = list(set(candidate_point_list) - set(temp_list))

#             cluster_center_point_list = list(set(cluster_center_point_list) | 
#                                              ( set(temp_list) & set(center_point_list)) )
#             cluster_center_point_list.remove(cluster_center_point)

#         # 步骤5续
#         cluster_list.append(cluster)
#         center_point_list = list(set(center_point_list) - set(cluster))

#     # 在DBSCAN中，单个点为噪声点，可不进行处理
    
#     # # 判断簇分划是否完整，如有缺少则添加单个点
#     # flag_list = [False] * len(state_dict_list)

#     # for cluster in cluster_list:
#     #     if cluster is list:
#     #         for point in cluster:
#     #             flag[point] = True
#     #     else:
#     #         flag[cluster] = True

#     # for flag in flag_list:
#     #     if flag == False:
#     #         cluster_list.append([flag_list.index(flag)])
    
#     return cluster_list

    
