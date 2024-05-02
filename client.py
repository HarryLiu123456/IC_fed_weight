import os
import csv

import torch

import model

class Client:

    def __init__(self, client_id, model_name, dataset, device, train_ratio, batch_size, class_num, class_list):
        
        # 初始化
        self.client_id = client_id
        self.class_list = class_list
        print(f'Client id: {self.client_id}, ' +
              f'Local classes: {set(self.class_list)}. ')
        
        # 目录与记录文件生成
        self.log_dir = f'log/client_{client_id}'
        os.makedirs(self.log_dir)

        with open( f'{self.log_dir}/log.csv' , 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Client_id', 'Global_epoch_id', 'Local_epoch_id', 'Local_acc'])

        # 用于训练的模型
        self.local_model = model.get_model(model_name, class_num)
        self.local_model.to(device)

        # 用于评估效果的模型
        self.eval_model = model.get_model(model_name, class_num)
        self.eval_model.to(device)

        # 局部数据集分割
        data_num = len(dataset)
        train_num = int(data_num * train_ratio)
        self.train_set, self.val_set = torch.utils.data.random_split(dataset,
                                                                   [train_num, data_num - train_num])
        self.train_loader = torch.utils.data.DataLoader(    self.train_set, 
                                                            batch_size=batch_size,
                                                            shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(  self.val_set,
                                                        batch_size=batch_size,
                                                        shuffle=False)
    
    # 权值归还
    def param_return(self, update_state_dict):
        self.local_model.load_state_dict(update_state_dict)
    
    # 局部训练
    def local_train(self, local_epoch_num, device, global_epoch_id, lr, momentum):
        
        # 首次评估
        self.local_model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for idx, data in enumerate(self.val_loader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self.local_model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(dim=0)
                correct += (predicted == labels).sum().item()

        print(f'Client id: {self.client_id}, ' +
                f'Global epoch id: {global_epoch_id}, ' + 
                f'Local epoch id: -1, ' +
                f'Local acc: {100 * correct / total:.1f}%. ')
        
        with open( f'{self.log_dir}/log.txt' , 'a') as f:
            f.write(f'Client id: {self.client_id}, ' +
                f'Global epoch id: {global_epoch_id}, ' + 
                f'Local epoch id: -1, ' +
                f'Local acc: {100 * correct / total:.1f}%. \n')
            
        with open( f'{self.log_dir}/log.csv' , 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.client_id, global_epoch_id, '-1', 100 * correct / total])

        optimizer = torch.optim.SGD(self.local_model.parameters(), 
                                    lr=lr,
									momentum=momentum)
        
        for local_epoch_id in range(local_epoch_num):
            
            # 循环训练
            self.local_model.train()

            running_loss = 0.0
            for idx, data in enumerate(self.train_loader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.local_model(images)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

                # .item方法由单元素张量返回标量
                running_loss += loss.item()

            print(f'Client id: {self.client_id}, ' + 
                  f'Global epoch id: {global_epoch_id}, ' + 
                  f'Local epoch id: {local_epoch_id}, ' + 
                  f'Loss: {running_loss / len(self.train_loader):.3f}. ')
            
            # 循环评估
            self.local_model.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for idx, data in enumerate(self.val_loader):
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = self.local_model(images)
                    _, predicted = torch.max(outputs, 1)

                    total += labels.size(dim=0)
                    correct += (predicted == labels).sum().item()

            print(f'Client id: {self.client_id}, ' +
                  f'Global epoch id: {global_epoch_id}, ' + 
                  f'Local epoch id: {local_epoch_id}, ' +
                  f'Local acc: {100 * correct / total:.1f}%. ')
            
            with open( f'{self.log_dir}/log.txt' , 'a') as f:
                f.write(f'Client id: {self.client_id}, ' +
                        f'Global epoch id: {global_epoch_id}, ' + 
                        f'Local epoch id: {local_epoch_id}, ' +
                        f'Local acc: {100 * correct / total:.1f}%. \n')
                
            with open( f'{self.log_dir}/log.csv' , 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.client_id, global_epoch_id, local_epoch_id, 100 * correct / total])

        # 状态字典
        state_dict = self.local_model.state_dict()

        return state_dict
    
    # 评估别人的权值，返回准确率
    def local_eval(self, device, state_dict):

        self.eval_model.load_state_dict(state_dict)

        self.eval_model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for idx, data in enumerate(self.val_loader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self.eval_model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(dim=0)
                correct += (predicted == labels).sum().item()
                
        return correct / total


