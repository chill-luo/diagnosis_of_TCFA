#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import torch
import numpy as np


#------------------------------------------------------------------------------
#   Fundamental metrics
#------------------------------------------------------------------------------
def miou(outputs, targets, eps=1e-6):
    """
    outputs: (torch.float32)  shape (N, C, H, W)
    targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
    """
    outputs = torch.argmax(outputs, dim=1, keepdim=True).type(torch.float32)
    outputs = outputs.squeeze()

    inter = torch.sum((outputs == targets) * (targets > 0))
    output = (outputs > 0).type(torch.float32)
    target = (targets > 0).type(torch.float32)

    inter_1 = torch.dot(output.view(-1), target.view(-1))
    union = torch.sum(output) + torch.sum(target) - inter_1 + eps

    iou = inter / (union + eps)
    return iou.mean()


#------------------------------------------------------------------------------
#   Custom metrics
#------------------------------------------------------------------------------
def sub_iou(num, outputs, targets, argmax=True, eps=1e-6):
    """
    outputs: (torch.float32)  shape (N, C, H, W)
    targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
    """
    if argmax:
        outputs = torch.argmax(outputs, dim=1, keepdim=True).type(torch.float32)
    outputs = outputs.squeeze(0)

    output = (outputs == num).type(torch.float32)
    target = (targets == num).type(torch.float32)
    if torch.sum(target) == 0:
        output = 1 - output
        target = 1 - target
    inter = torch.dot(output.view(-1), target.view(-1))
    union = torch.sum(output) + torch.sum(target) - inter + eps

    iou = inter / (union + eps)
    return iou.mean()


def iou_for_lip(outputs, targets, argmax=True, eps=1e-6):
	"""
	logits: (torch.float32)  shape (N, C, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	if argmax:
		outputs = torch.argmax(outputs, dim=1, keepdim=True).type(torch.int64)
	targets = torch.unsqueeze(targets, dim=1).type(torch.int64)

	region_out = (outputs==2).type(torch.int64).max(dim=3).values
	region_tar = (targets==2).type(torch.int64).max(dim=3).values

	inter = (region_out & region_tar).type(torch.float32).sum(dim=2)
	union = (region_out | region_tar).type(torch.float32).sum(dim=2)

	iou = inter / (union + eps)
	return iou.mean()


def miou_sp(outputs, targets):
	"""
    outputs: (torch.float32)  shape (N, C, H, W)
    targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
    """
	n_class = outputs.shape[1]
	mious = []
	for c in range(1, n_class):
		if c == 2:
			iou = iou_for_lip(outputs, targets)
		else:
			iou = sub_iou(c, outputs, targets)
		mious.append(iou.mean())
	return torch.FloatTensor(mious)