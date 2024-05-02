import torch
import torch.nn as nn
import torchvision.ops as ops
from scipy.optimize import linear_sum_assignment
from args import device


def IOU(bbox1, bbox2):
    bbox1 = ops.box_convert(bbox1.reshape(-1, 4), in_fmt='cxcywh', out_fmt='xyxy')  # [B*N,4]
    bbox2 = ops.box_convert(bbox2.reshape(-1, 4), in_fmt='cxcywh', out_fmt='xyxy')  # [B*N,4]
    iou = torch.diagonal(ops.box_iou(bbox1, bbox2))  # [B*N]
    diou_loss = ops.distance_box_iou_loss(bbox1, bbox2)  # [B*N]
    return iou, diou_loss


def calculate_cost_matrix(labels_pred, bbox_pred, labels_target, bbox_target):
    # labels_pred.shape = (B,N), bbox_pred.shape = (B,N,4)
    # labels_target.shape = (B,M), bbox_target.shape = (B,M,4)
    B, N = labels_pred.shape[0:2]
    B, M = labels_target.shape[0:2]
    cost_matrix = torch.zeros((B, N, M)).to(device)
    for i in range(N):
        for j in range(M):
            diou_cost = IOU(bbox_pred[:, i:i + 1, :], bbox_target[:, j:j + 1, :])[1]  # (B,1)
            label_cost = nn.functional.binary_cross_entropy(labels_pred[:, i:i + 1], labels_target[:, j:j + 1])  # (B,1)
            bbox_cost = nn.functional.l1_loss(bbox_pred[:, i:i + 1, :], bbox_target[:, j:j + 1, :])  # (B,1)
            cost_matrix[:, i, j] = label_cost.squeeze() + diou_cost.squeeze() + bbox_cost.squeeze()
    row_indices = []
    col_indices = []
    for batch_idx in range(B):
        cost_matrix_np = cost_matrix[batch_idx].detach().cpu().numpy()
        row_idx, col_idx = linear_sum_assignment(cost_matrix_np)
        row_indices.append(torch.tensor(row_idx, dtype=torch.long))
        col_indices.append(torch.tensor(col_idx, dtype=torch.long))
    row_indices = torch.stack(row_indices, dim=0)
    col_indices = torch.stack(col_indices, dim=0)
    return cost_matrix, row_indices, col_indices


class Loss_fn(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, labels_pred, bbox_pred, labels_target, bbox_target):
        B = labels_pred.shape[0]
        labels_pred = torch.sigmoid(labels_pred).squeeze(2)
        cost_matrix, row_indices, col_indices = calculate_cost_matrix(labels_pred, bbox_pred, labels_target,
                                                                      bbox_target)
        batch_indices = torch.arange(cost_matrix.size(0)).view(-1, 1).expand_as(row_indices)

        labels_pred = labels_pred[batch_indices, row_indices]
        bbox_pred = bbox_pred[batch_indices, row_indices, :]
        labels_target = labels_target[batch_indices, col_indices]
        bbox_target = bbox_target[batch_indices, col_indices]
        mask = (labels_target == 1)
        bbox_pred = bbox_pred[mask]
        bbox_target = bbox_target[mask]

        diou_loss = IOU(bbox_pred, bbox_target)[1].mean()
        iou = IOU(bbox_pred, bbox_target)[0].mean()
        label_loss = nn.functional.binary_cross_entropy(labels_pred, labels_target)
        bbox_loss = nn.functional.l1_loss(bbox_pred, bbox_target)
        total_loss = diou_loss + label_loss + bbox_loss
        print('diou_loss:', diou_loss.item(), 'label_loss:', label_loss.item(), 'bbox_loss:', bbox_loss.item())
        print('iou:', iou.item(), 'total_loss:', total_loss.item())

        return total_loss, iou
