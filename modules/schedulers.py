"""
Implements learning rate schedulers for optimizing model training in PyTorch.  
Includes custom implementations to control learning rate decay.  
Future updates may add support for task-specific scheduling strategies.
"""

import math
import torch
from torch.optim.lr_scheduler import LambdaLR

class FixedLRSchedulerWrapper:

    """
    FixedLRSchedulerWrapper wraps an existing learning rate scheduler to allow for 
    a fixed learning rate after a specified epoch.

    Args:
        scheduler (torch.optim.lr_scheduler): The underlying learning rate scheduler.
        fixed_lr (float): The learning rate to be fixed after `fixed_epoch`.
        fixed_epoch (int): The epoch after which the learning rate should be fixed.

    Attributes:
        scheduler (torch.optim.lr_scheduler): The underlying scheduler used for learning rate adjustment.
        fixed_lr (float): The fixed learning rate value.
        fixed_epoch (int): The epoch after which the fixed learning rate is applied.
        last_epoch (int): Tracks the last epoch processed by the scheduler.
    """

    def __init__(self, scheduler, fixed_lr, fixed_epoch):

        """
        Initializes the custom scheduler with the given scheduler, fixed learning rate,
        and fixed epoch.

        Args:
            scheduler (torch.optim.lr_scheduler): The original scheduler (e.g., CosineAnnealingLR).
            fixed_lr (float): The fixed learning rate after the specified epoch.
            fixed_epoch (int): The epoch at which to stop using the original scheduler
                               and switch to the fixed learning rate.
        """

        self.scheduler = scheduler
        self.fixed_lr = fixed_lr
        self.fixed_epoch = fixed_epoch 
        self.last_epoch = 0

    def step(self, epoch=None):

        """
        Updates the learning rate based on the current epoch. Uses the original scheduler
        before `fixed_epoch` and applies a fixed learning rate afterward.

        Args:
            epoch (int, optional): The current epoch. If provided, updates the internal epoch counter.
        """

        # Update the current epoch if provided
        if epoch is not None:
            self.last_epoch = epoch
        else:
            self.last_epoch += 1
        
        # Delegate to the original scheduler if before the fixed epoch
        if self.last_epoch < self.fixed_epoch:
            
            self.scheduler.step(epoch)

        # Set the learning rate to the fixed value for all parameter groups
        else:
            for param_group in self.scheduler.optimizer.param_groups:
                param_group['lr'] = self.fixed_lr

    def get_last_lr(self):
        """
        Retrieves the current learning rate. Uses the original scheduler's learning rate
        before `fixed_epoch` and returns the fixed learning rate afterward.

        Returns:
            list: The current learning rate(s) as a list (one per parameter group).
        """
        if self.last_epoch < self.fixed_epoch:
            return self.scheduler.get_last_lr()  # Delegate to the original scheduler
        else:
            return [self.fixed_lr]  # Return the fixed learning rate
