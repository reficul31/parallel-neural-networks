from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts

def get_scheduler(scheduler, optimizer):
    if scheduler == 'warm':
        return CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)
    elif scheduler == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6, last_epoch=-1)
    elif scheduler == 'reduce':
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    else:
        return None