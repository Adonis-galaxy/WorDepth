import torch
import torch.nn as nn


class SILogLoss(nn.Module):
    def __init__(self, SI_loss_lambda, max_depth):
        '''
        Scale Invariant loss for training

        Arg(s):
            SI_loss_lambda: float
                weight for the scale invariant term
            max_depth: float
                maximum depth for calculating loss
        '''
        super(SILogLoss, self).__init__()

        self.SI_loss_lambda = SI_loss_lambda
        self.max_depth = max_depth

    def forward(self, depth_pred, depth_gt):
        '''
        Calculate SILogloss

        Arg(s):
            depth_pred: torch.Tensor[float32]
                N x 1 x Height (480 for nyu) x Width (640 for nyu)
            depth_gt: torch.Tensor[float32]
                N x 1 x Height (480 for nyu) x Width (640 for nyu)
        Returns:
            loss: float
        '''

        loss = 0

        diff = torch.log(depth_pred) - torch.log(depth_gt)

        num_pixels = (depth_gt > 0.1) * (depth_gt < self.max_depth)

        diff = torch.where(
            (depth_gt > 0.1) * (depth_gt < self.max_depth) * (torch.abs(diff) > 0.001),
            diff,
            torch.zeros_like(diff)
        )
        lamda = self.SI_loss_lambda

        diff = diff.reshape(diff.shape[0], -1)
        num_pixels = num_pixels.reshape(num_pixels.shape[0], -1).sum(dim=-1) + 1e-6

        loss1 = (diff**2).sum(dim=-1) / num_pixels
        loss1 = loss1 - lamda * (diff.sum(dim=-1) / num_pixels) ** 2

        total_pixels = depth_gt.shape[1] * depth_gt.shape[2] * depth_gt.shape[3]

        weight = num_pixels.to(diff.dtype) / total_pixels

        loss = (loss1 * weight).sum()

        return loss
