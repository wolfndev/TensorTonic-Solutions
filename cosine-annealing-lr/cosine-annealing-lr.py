import numpy as np
def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    """
    Compute the learning rate using cosine annealing.
    """
    # Write code here
    if current_step == 0:
        return base_lr
    elif current_step == total_steps:
        return min_lr
    else:
        return min_lr + 1/2*(base_lr - min_lr)*(1 + np.cos(np.pi*current_step/total_steps))