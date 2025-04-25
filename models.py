import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
import math

class MLP(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=128, out_dim=10, depth=1, bn=True, init_type='kaiming'):
        super(MLP, self).__init__()


        block = lambda inp, out: nn.Sequential(
            nn.Linear(inp, out),
            nn.BatchNorm1d(out) if bn else nn.Identity(),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.head = block(input_dim, hidden_dim)

        self.body = nn.Sequential(*[block(hidden_dim, hidden_dim) for _ in range(depth)])

        self.tail = nn.Linear(hidden_dim, out_dim)

        # Initialize (body?) layers
        self._initialize_weights(init_type)

    def _initialize_weights(self, init_type):
        init_fncs = {
            'kaiming': partial(nn.init.kaiming_normal_, nonlinearity='relu'),
            'orthogonal': partial(nn.init.orthogonal_, gain = math.sqrt(2.0)), # Recommended gain for ReLUs
        }
        for m in self.body.modules():
            if isinstance(m, nn.Linear):
                init_fncs[init_type](m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.flatten(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x
    

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in,  channels_out, 3, padding='same', bias=False)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = nn.BatchNorm2d(channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding='same', bias=False)
        self.norm2 = nn.BatchNorm2d(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

class VanillaCNN(nn.Module):
    def __init__(self, out_dim=10, init_type='kaiming', num_blocks=3, base_width=64, widen_factor=2):
        super().__init__()

        # First "whitening" conv
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2

        layers = [
            nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
            nn.GELU()
        ]

        # ConvGroups with increasing widths
        in_channels = whiten_width
        for i in range(num_blocks):
            out_channels = base_width * (widen_factor ** i)
            layers.append(ConvGroup(in_channels, out_channels))
            in_channels = out_channels

        # Final pool to shrink spatial size
        layers.append(nn.MaxPool2d(3))

        self.body = nn.Sequential(*layers)

        # Tail assumes small spatial dim after ConvGroups + MaxPool
        self.tail = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, out_dim, bias=False),
        )

        self._initialize_weights(init_type)
    
    def _initialize_weights(self, init_type):
        init_fncs = {
            'kaiming': partial(nn.init.kaiming_normal_, nonlinearity='relu'),
            'orthogonal': partial(nn.init.orthogonal_, gain=math.sqrt(2.0)),
        }
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_fncs[init_type](m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init_fncs[init_type](m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.body(x)
        x = self.tail(x)
        return x
