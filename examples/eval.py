from efficientnet_pytorch import EfficientNet

'''
计算efficentnet的参数量
'''
model = EfficientNet.from_pretrained('efficientnet-b0')
print('# generator parameters:', sum(param.numel() for param in model.parameters()))
