import os
import csv

import torch

import model

class Server(object):

	def __init__(self, model_name, dataset, device, batch_size, class_num):

		# 目录与日志初始化
		self.log_dir = 'log/server'
		os.makedirs(self.log_dir)
		
		with open( f'{self.log_dir}/log.csv' , 'a', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(['Global_epoch_id', 'Global_acc'])

		# 全局模型
		self.global_model = model.get_model(model_name, class_num)
		self.global_model.to(device)

		# 数据加载器
		self.val_loader = torch.utils.data.DataLoader(	dataset, 
												 		batch_size=batch_size, 
												 		shuffle=False)
	
	# 模型整合与分配
	def model_integrate_allocate(self, state_dict_list, final_weight):

		update_state_dict_list = []

		weight_accumulator = {}
		for name, param in self.global_model.state_dict().items():
			# torch.zeros_like默认存储类型与输入一致
			weight_accumulator[name] = torch.zeros_like(param)

		for idx, state_dict in enumerate(state_dict_list):
			for name, param in state_dict.items():
				temp_vector = param * final_weight[idx]
				weight_accumulator[name] += temp_vector.to(param.dtype)
				weight_accumulator[name].to(param.dtype)

		# TODO: 不确定clone方法是否必要
		for name, param in self.global_model.state_dict().items():
			# if param.type() != weight_accumulator[name].type():
			# 	param.copy_(weight_accumulator[name].to(torch.int64).clone())
			# else:
			# 	param.copy_(weight_accumulator[name].clone())
			param.copy_(weight_accumulator[name].clone())

		for _ in range(len(state_dict_list)):
			update_state_dict_list.append(self.global_model.state_dict())

		return update_state_dict_list
		
	# 全局模型评估	
	def global_eval(self, device, global_epoch_id):

		self.global_model.eval()

		correct = 0
		total = 0

		with torch.no_grad():
			for idx, data in enumerate(self.val_loader):
				images, labels = data 
				images, labels = images.to(device), labels.to(device)
				outputs = self.global_model(images)
				_, predicted = torch.max(outputs, 1)

				total += labels.size(dim=0)
				correct += (predicted == labels).sum().item()

		print(f'Global epoch id: {global_epoch_id}, ' + 
                f'Global acc: {100 * correct / total:.1f}%. ')
		
		with open( f'{self.log_dir}/log.txt' , 'a') as f:
			f.write(f'Global epoch id: {global_epoch_id}, ' + 
					f'Global acc: {100 * correct / total:.1f}%. \n')
			
		with open( f'{self.log_dir}/log.csv' , 'a', newline='') as f:
			writer = csv.writer(f)
			writer.writerow([global_epoch_id, 100 * correct / total])

