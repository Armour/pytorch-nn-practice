
# coding: utf-8

# In[1]:


import math

import torch.nn as nn


# In[2]:


cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    """
    VGG network template
    """
    def __init__(self, vgg_name, batch_norm=True, num_classes=1000):
        """
        Init vgg network
        Args:
            vgg_name (string): VGG identifier, map to different config parameters 
            batch_norm (bool): If True, use batch normalization
            num_classes (int): The number of output classes
        """
        super(VGG, self).__init__()
        if vgg_name in cfg:
            self.features = self._make_layers(cfg[vgg_name], batch_norm)
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096), 
                nn.ReLU(inplace=True), 
                nn.Dropout(),
                nn.Linear(4096, 4096), 
                nn.ReLU(inplace=True), 
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
            self._initialize_weights()
        else:
            print('vgg_name doesn\'t exist, please choose from vgg11, vgg13, vgg16 and vgg19')

    def forward(self, x):
        """
        Pytorch forward function
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """
        Init weight parameters
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _make_layers(self, cfg, batch_norm):
        """
        Generate network layers based on configs
        """
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
                
# net = VGG('vgg16', num_classes=10)
# print(net)

