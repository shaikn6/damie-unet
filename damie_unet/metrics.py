"""Dataset-level metrics. Rows are truth, columns are predictions."""

import torch


class ConfusionMatrix:
    def __init__(self):
        self.matrix = None

    def update(self, prediction, target):
        prediction = prediction.detach().to(torch.int64).flatten()
        target = target.detach().to(torch.int64).flatten()
        if ((target < 0) | (target > 1) | (prediction < 0) | (prediction > 1)).any():
            raise ValueError('labels must be 0 (ice) or 1 (pond)')
        # Four masked sums instead of torch.bincount, which is ~330 ms/image on MPS.
        code = target * 2 + prediction
        counts = torch.stack([(code == i).sum() for i in range(4)])
        if self.matrix is None:
            self.matrix = torch.zeros((2, 2), dtype=torch.int64, device=counts.device)
        self.matrix += counts.reshape(2, 2)

    def compute(self):
        counts = self.matrix.cpu() if self.matrix is not None else torch.zeros((2, 2), dtype=torch.int64)
        matrix = counts.double()
        true_positive = matrix.diag()
        truth, predicted = matrix.sum(1), matrix.sum(0)
        # An absent class has IoU zero and participates in the two-class mean.
        iou = true_positive / (truth + predicted - true_positive).clamp_min(1)
        f1 = 2 * true_positive / (truth + predicted).clamp_min(1)
        return {'iou': iou.tolist(), 'miou': iou.mean().item(),
                'f1_pond': f1[1].item(), 'f1_macro': f1.mean().item(),
                'precision_pond': (true_positive[1] / predicted[1].clamp_min(1)).item(),
                'recall_pond': (true_positive[1] / truth[1].clamp_min(1)).item(),
                'accuracy': (true_positive.sum() / matrix.sum().clamp_min(1)).item(),
                'confusion_matrix': counts.tolist()}


def pond_iou_per_image(predictions, masks):
    """Pond IoU of each image in a batch; a scene with no true or predicted pond scores zero,
    the same rule as the pooled metric."""
    pond_prediction, pond_truth = predictions == 1, masks == 1
    intersection = (pond_prediction & pond_truth).flatten(1).sum(1)
    union = (pond_prediction | pond_truth).flatten(1).sum(1)
    return (intersection / union.clamp_min(1)).tolist()
