import torch 
import torchvision

# 得到实例并修改全连接层
def get_model(model_name, class_num):
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet34':
        model = torchvision.models.resnet34(weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, class_num, dtype=model.fc.weight.dtype)
    return model