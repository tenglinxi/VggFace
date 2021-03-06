import os
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchsummary import summary
import torchvision.datasets as dataset
import torch.nn as nn
from collections import OrderedDict


class VGGFace(nn.Module):
    def __init__(self):
        super(VGGFace, self).__init__()

        self.features = nn.ModuleDict(OrderedDict(
            {
                # === Block 1 ===
                'conv_1_1': nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                'relu_1_1': nn.ReLU(inplace=True),
                'conv_1_2': nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                'relu_1_2': nn.ReLU(inplace=True),
                'maxp_1_2': nn.MaxPool2d(kernel_size=2, stride=2),
                # === Block 2 ===
                'conv_2_1': nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                'relu_2_1': nn.ReLU(inplace=True),
                'conv_2_2': nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                'relu_2_2': nn.ReLU(inplace=True),
                'maxp_2_2': nn.MaxPool2d(kernel_size=2, stride=2),
                # === Block 3 ===
                'conv_3_1': nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                'relu_3_1': nn.ReLU(inplace=True),
                'conv_3_2': nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                'relu_3_2': nn.ReLU(inplace=True),
                'conv_3_3': nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                'relu_3_3': nn.ReLU(inplace=True),
                'maxp_3_3': nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                # === Block 4 ===
                'conv_4_1': nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                'relu_4_1': nn.ReLU(inplace=True),
                'conv_4_2': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_4_2': nn.ReLU(inplace=True),
                'conv_4_3': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_4_3': nn.ReLU(inplace=True),
                'maxp_4_3': nn.MaxPool2d(kernel_size=2, stride=2),
                # === Block 5 ===
                'conv_5_1': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_5_1': nn.ReLU(inplace=True),
                'conv_5_2': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_5_2': nn.ReLU(inplace=True),
                'conv_5_3': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_5_3': nn.ReLU(inplace=True),
                'maxp_5_3': nn.MaxPool2d(kernel_size=2, stride=2)
            }))

        self.fc = nn.ModuleDict(OrderedDict(
            {
                'fc6': nn.Linear(in_features=512 * 7 * 7, out_features=4096),
                'fc6-relu': nn.ReLU(inplace=True),
                'fc6-dropout': nn.Dropout(p=0.5),
                'fc7': nn.Linear(in_features=4096, out_features=4096),
                'fc7-relu': nn.ReLU(inplace=True),
                'fc7-dropout': nn.Dropout(p=0.5),
                'fc8': nn.Linear(in_features=4096, out_features=5737),
            }))

    def forward(self, x):
        # Forward through feature layers
        for k, layer in self.features.items():
            x = layer(x)

        # Flatten convolution outputs
        x = x.view(x.size(0), -1)

        # Forward through FC layers
        for k, layer in self.fc.items():
            x = layer(x)

        return x

if __name__ == '__main__':

    '''???????????????'''
    batch_size = 32  # ????????????
    num_epoches = 10  # ????????????????????????

    dataset = dataset.ImageFolder(
        root="lfw",
        transform=transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) # ????????????????????????

    '''??????model??????????????????????????????????????????GPU'''
    model = VGGFace()  # ??????????????????
    summary(model, input_size=(3, 224, 224), batch_size=batch_size, device='cpu')  # ??????????????????

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    '''?????? loss ??? optimizer '''
    loss_func = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    '''   ????????????
    - ???????????????loss = loss_func(out,batch_y)
    - ????????????????????????????????????opt.zero_grad()
    - ?????????????????????loss.backward()
    - ???????????????????????????net???parmeters??????opt.step()
    '''
    for epoch in range(num_epoches):
        model.train()
        print('\n', '*' * 25, 'epoch {}'.format(epoch + 1), '*' * 25)  # .format??????????????????formet??????????????????????????????????????????
        running_loss = 0.0
        num_correct = 0.0
        for i, data in enumerate(dataloader, 0):
            img, label = data
            img, label = img.to(device), label.to(device)  # ????????????Tensor, ?????? Variable

            out = model(img)  # ????????????

            # ????????????
            loss = loss_func(out, label)  # ??????loss
            optimizer.zero_grad()  # ????????????????????????????????????
            loss.backward()  # loss ??????, ??????????????????????????????????????????
            optimizer.step()  # ??????????????????????????????????????????net???parmeters???

            # ??????loss ??? acc
            running_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)  # ????????????????????????????????????
            num_correct += (pred == label).sum().item()  # ?????????????????????
            # print('==> epoch={}, running_loss={}, num_correct={}'.format(i+1, running_loss, num_correct))

        print(
            'Train==> Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, running_loss / (len(dataloader)), num_correct / (len(dataloader))))

        # ?????? ?????? ??????
        model.eval()  # ?????????????????????????????????????????????
        eval_loss = 0
        num_correct = 0
        for data in dataloader:  # ????????????
            img, label = data
            img, label = img.to(device).detach(), label.to(device).detach()  # ????????????????????????

            out = model(img)
            loss = loss_func(out, label)
            eval_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct += (pred == label).sum().item()
        print('Test==>  Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(dataloader)), num_correct / (len(dataloader))))

    # ????????????
    torch.save(model.state_dict(), 'VGGNet16_cifar10.pth')
