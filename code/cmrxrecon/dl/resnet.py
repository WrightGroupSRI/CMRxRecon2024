from torch import nn

class residual_block(nn.Module):
    def __init__(self, chans, scaling, dimensions='2d') -> None:
        super().__init__()
        
        if dimensions == '2d':
            conv = nn.Conv2d
            instance_norm = nn.InstanceNorm2d
        else: 
            conv = nn.Conv1d
            instance_norm = nn.InstanceNorm1d

        self.res_block = nn.Sequential(
                conv(chans, chans, 3, stride=1, padding=1, bias=False),
                instance_norm(chans),
                nn.ReLU(),
                conv(chans, chans, 3, stride=1, padding=1, bias=False),
        )
        self.scaling = scaling

    def forward(self, x):
        return self.res_block(x) * self.scaling + x

class ResNet(nn.Module):
    def __init__(self, itterations=10, in_chan=2, out_chan=2, chans=32, scaling=0.1, dimension='2d') -> None:
        super().__init__()

        self.cascade = nn.Sequential()
        for _ in range(itterations):
            self.cascade.append(residual_block(chans, scaling, dimension))
        
        if dimension == '2d': 
            self.cascade.append(nn.Conv2d(chans, chans, 3, padding=1, bias=False))
            self.encode = nn.Conv2d(in_chan, chans, 3, padding=1, bias=False)
            self.decode = nn.Conv2d(chans, out_chan, 3, padding=1, bias=False)
        else:
            self.cascade.append(nn.Conv1d(chans, chans, 3, padding=1, bias=False))
            self.encode = nn.Conv1d(in_chan, chans, 3, padding=1, bias=False)
            self.decode = nn.Conv1d(chans, out_chan, 3, padding=1, bias=False)
    

    def forward(self, x):
        x = self.encode(x)
        x = self.cascade(x) + x
        x = self.decode(x)
        return x

