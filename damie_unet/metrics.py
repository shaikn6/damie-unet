"""Dataset-level metrics. Rows are truth, columns are predictions."""

import torch


class ConfusionMatrix:
    def __init__(self):
        self.matrix = None

    def update(self, prediction, target):
        prediction = prediction.detach().to(torch.int64).flatten()
        target = target.detach().to(torch.int64).flatten()
        counts = torch.bincount(target * 2 + prediction, minlength=4)
        if counts.numel() != 4:
            raise ValueError('labels must be 0 (ice) or 1 (pond)')
        if self.matrix is None:
            self.matrix = torch.zeros((2, 2), dtype=torch.int64, device=counts.device)
        self.matrix += counts.reshape(2, 2)

    def compute(self):
        counts = self.matrix.cpu() if self.matrix is not None else torch.zeros((2, 2), dtype=torch.int64)
        matrix = counts.double()
        true_positive = matrix.diag()
        union = matrix.sum(0) + matrix.sum(1) - true_positive
        # An absent class has IoU zero and participates in the two-class mean.
        iou = true_positive / union.clamp_min(1)
        f1 = 2 * matrix[1, 1] / (matrix[1].sum() + matrix[:, 1].sum()).clamp_min(1)
        return {'iou': iou.tolist(), 'miou': iou.mean().item(),
                'f1_pond': f1.item(), 'confusion_matrix': counts.tolist()}
