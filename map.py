import torch 
from iou import intersection_over_union

def non_max_suppression(bboxes, iou_threshold, threshold, box_format="midpoint"):
    """
    Mengurangi kotak prediksi dengan teknik non max suppression
    
    Parameters:
        bboxes (list): [[class_pred, prob_score, x, y, width, height]]
        iou_threshold (float): threshold dimana kotak prediksi berdekatan
        threshold (float): threshold untuk menghapus kotak prekdisi
        box_format (str): midpoint atau corners
        
    Returns:
        list: kotak prediksi setelah dikenakan non max suppression dengan iou threshold tertentu
    """
    
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x : x[1], reverse=True)
    bboxes_after_nms = []
    
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0] # Kelasnya berbeda
            or intersection_over_union(
                torch.tensor(box[2:]),
                torch.tensor(chosen_box[2:]),
                box_format=box_format
            ) < iou_threshold # kotak prediksi yang tidak berdekatan
        ]
        
        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms