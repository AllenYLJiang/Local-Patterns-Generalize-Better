import torch
inp = torch.randn(256, 24, 36, dtype=torch.float64)
input_frame = 20
pred_t = inp[:,input_frame:24,:].reshape(-1, (24-input_frame )* 36)
mse = torch.nn.MSELoss(reduce=False)
lso = mse(torch.randn(256, 144), pred_t).mean(dim=1)
print()
lso.float()
print(lso.dtype)