import torch
import torch.nn.functional as F

''' endpoint error '''
def lossEPE(output, target):
  b, _, h, w = target.size()
  upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
  return torch.norm(target - upsampled_output, 1, 1).mean()
