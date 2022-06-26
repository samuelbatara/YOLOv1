import torch

def intersection_over_union(pred_bboxes, true_bboxes, box_format="midpoint"):
    """
    Menghitung intersection over union
    
    Parameters:
        boxes_preds (tensor): Kotak prediksi (batch_size, 4)
        boxes_labels (tensor): Kotak label yang benar (batch_size, 4)
        box_format: midpoint atau corners
        
    Returns:
        tensor: nilai intersection over union
    """
    
    if box_format == "midpoint":
        box1_x1 = pred_bboxes[..., 0:1] - pred_bboxes[..., 2:3]/2
        box1_y1 = pred_bboxes[..., 1:2] - pred_bboxes[..., 3:4]/2
        box1_x2 = pred_bboxes[..., 0:1] + pred_bboxes[..., 2:3]/2
        box1_y2 = pred_bboxes[..., 1:2] + pred_bboxes[..., 3:4]/2
        box2_x1 = true_bboxes[..., 0:1] - true_bboxes[..., 2:3]/2
        box2_y1 = true_bboxes[..., 1:2] - true_bboxes[..., 3:4]/2
        box2_x2 = true_bboxes[..., 0:1] + true_bboxes[..., 2:3]/2
        box2_y2 = true_bboxes[..., 1:2] + true_bboxes[..., 3:4]/2
    elif box_format == "corners":
        box1_x1 = pred_bboxes[..., 0:1]
        box1_y1 = pred_bboxes[..., 1:2]
        box1_x2 = pred_bboxes[..., 2:3]
        box1_y2 = pred_bboxes[..., 3:4]
        box2_x1 = true_bboxes[..., 0:1]
        box2_y1 = true_bboxes[..., 1:2]
        box2_x2 = true_bboxes[..., 2:3]
        box2_y2 = true_bboxes[..., 3:4]
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    
    return intersection / (box1_area + box2_area - intersection + 1e-6)