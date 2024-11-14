import torch

office0 = torch.load('/workspace/data/replica_processed/replica_3d/office0.pth')
for i in office0:
    print(i.shape)