import torch
from torchvision import models
from thop import profile
from thop import clever_format
from models import WSDAN_MCS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = models.inception_v3(pretrained=True).to(device)
net = WSDAN_MCS(num_classes=200, M=32, net='resnet101', pretrained=True)
# checkpoint = torch.load('./FGVC/CUB-200-2011/ckpt/v5.pth')  #resnet101.ckpt
# state_dict = checkpoint['state_dict']
# net.load_state_dict(state_dict)
net.to(device)

input = torch.randn(1, 3, 448, 448).to(device)
macs, params = profile(net, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
print(f"FLOPs: {macs} GFLOPs")
print(f"Parameters: {params} M")