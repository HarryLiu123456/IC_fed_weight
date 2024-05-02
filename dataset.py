import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_dataset(dataset_name):
    if dataset_name ==  "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        train_transform=transforms.Compose([        
                    # 先填充再裁剪，提高泛化性
                    transforms.RandomCrop(32, padding=4),
                    # 随机水平翻转
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # 归一化
                    transforms.Normalize(mean=mean,std=std)])
        val_transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean,std=std)])

        train_dataset=datasets.CIFAR10(
                            root='dataset',  
                            train=True,     
                            download=True,  
                            transform=train_transform
                        )
        val_dataset=datasets.CIFAR10(
                            root='dataset',  
                            train=False,     
                            download=True,  
                            transform=val_transform
                        )
    
    elif dataset_name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        train_transform=transforms.Compose([        
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean,std=std)])
        val_transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean,std=std)])
        train_dataset=datasets.CIFAR100(
                            root='dataset',  
                            train=True,     
                            download=True,  
                            transform=train_transform
                        )
        val_dataset=datasets.CIFAR100(
                            root='dataset',  
                            train=False,     
                            download=True,  
                            transform=val_transform
                        )
        
    return train_dataset, val_dataset




    



