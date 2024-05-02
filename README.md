* 本实验为 图像分类 下的联邦学习算法研究

* 赋权法
    * 主观赋权法
        * 层次分析法（Analytic Hierarchy Process,AHP）
            主要在主观评价，不适用
        * -
    * 客观赋权法
        * **<u>熵权法（Entropy Weight Method，EWM）</u>**
            各客户端数据集 与 各客户端模型权重
        * -

* 聚类
    * 划分式
        只能区分球形，不适用
        * k-means
        * k-means++
        * bi-kmeans
    * 基于密度
        任意形状
        * **<u>DBSCAN（Density-Based Spatial Clustering of Applications with Noise，具有噪声的基于密度的聚类方法）</u>**
            一个超参数
        * OPTICS
            多个超参数，运算量大，不适用
    * 层次化
        不适用

* 注意事项
    * 注释采用中文，输出一律英文
    * 在本项目熵权法中，客户端（对应数据集）即为指标（因为数据集有优劣差异），权重为方案（得到分数即为对应权重）
    * 本项目中，矩阵中一行代表一个客户端，一列代表一个权值
    * 运行main开始训练，运行plot产生图表（还未实现）

* 环境搭建
    * 如果有需要更换默认镜像源为官方源（清华源有些时候没有特定包）
    * 但是外网连接有问题，还是不建议
    ```
    conda config --set channel_alias https://conda.anaconda.org
    ```

    * linux cuda
    ```
    conda install python==3.10

    # CUDA 11.8
    conda install pytorch==2.2.0 torchvision==0.17.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    # CUDA 12.1
    conda install pytorch==2.2.0 torchvision==0.17.0 pytorch-cuda=12.1 -c pytorch -c nvidia
    
    conda install -c conda-forge scikit-learn
    ```

    * windows no-cuda
    ```
    conda install python==3.10
    conda install pytorch==2.2.0 torchvision==0.17.0 cpuonly -c pytorch
    conda install -c conda-forge scikit-learn
    ```

    * ipynb
    ```
    conda install ipykernel
    ```

* 模式
    * 'fedavg' 'fedavg_EWM' 'fedavg_EWM_DBSCAN'
    * 'all_data' 'order_split' 'random_split' 'class_split'
