import torch

# preprocessor
processor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_preprocessor')
# models
# V-JEPA 2
vjepa2_vit_large = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')