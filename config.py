from torch.utils.tensorboard import SummaryWriter

device = 'cuda'
IMG_SIZE = 64
batch_size = 64
lr = 3e-3
gamma = 0.7
seed = 142
num_classes = 101

writer = SummaryWriter('runs/ResNetSE')