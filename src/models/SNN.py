import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, encoding



class SNNNet(nn.Module):
    def __init__(self):
        super(SNNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.sn1 = neuron.LIFNode()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.sn2 = neuron.LIFNode()
        self.pool = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # 注意这里是 8*8
        self.sn3 = neuron.LIFNode()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x_seq):
        # x_seq shape: [T, batch, C, H, W]
        mem_out = 0
        for t in range(x_seq.shape[0]):
            x = self.conv1(x_seq[t])
            x = self.sn1(x)
            x = self.pool(x)

            x = self.conv2(x)
            x = self.sn2(x)
            x = self.pool(x)

            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.sn3(x)

            x = self.fc2(x)
            mem_out += x

        return mem_out / x_seq.shape[0]