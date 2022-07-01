import torch 
from collections import Counter
from iou import intersection_over_union
from merge_sort import merge_sort, left_bound

def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Menghitung mean average precision
    
    Parameters:
        pred_boxes (list): [[train_idx, class_pred, prob_score, x, y, width, height]]
        true_boxes (list): sama seperti pred_boxes
        iou_threshold (float): threshold dimana kotak prediksi benar
        box_format (str): midpoint atau corners
        num_classes (int): jumlah kelas
        
    Returns:
        float: nilai mAP untuk iou threshold tertentu
    """
    
    average_precisions = []
    epsilon = 1e-6
    
    # Mengurutkan pred_boxes and true_boxes menggunakan Merge Sort
    pred_boxes = merge_sort(pred_boxes, 0, len(pred_boxes)-1)
    true_boxes = merge_sort(true_boxes, 0, len(true_boxes)-1)
    
    for c in range(num_classes):
        "INGAT: index yang diberikan adalah lower bound"
        pred_lb = left_bound(pred_boxes, c)
        pred_rb = left_bound(pred_boxes, c+1)      
        true_lb = left_bound(true_boxes, c)
        true_rb = left_bound(true_boxes, c+1)
        
        "Jika true boxes kosong untuk kelas c,"
        "maka lanjut ke kelas selanjutnya"
        if(true_boxes[true_lb][1] != c):
            continue
        
        if(len(pred_boxes) and pred_boxes[pred_lb][1] == c):
            pred_rb = max(pred_lb, pred_rb-1)
            assert pred_boxes[pred_rb][1] == c, "Batas kanan dari kotak prediksi salah"
        if(len(true_boxes) and true_boxes[true_lb][1] == c):
            true_rb = max(true_lb, true_rb-1)
            assert true_boxes[true_rb][1] == c, "Batas kanan dari kotak target salah"
        
        temp = []
        for gt in true_boxes[true_lb:true_rb+1]:
            temp.append(int(gt[0]))
            assert(gt[1] == c)
        amount_bboxes = Counter(temp)
        
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
            
        TP = torch.zeros(pred_rb - pred_lb + 1)
        FP = torch.zeros(pred_rb - pred_lb + 1)
        total_true_boxes = true_rb - true_lb + 1
        
        if(len(pred_boxes) and pred_boxes[pred_lb][1] == c):
            for detection_idx, detection in enumerate(pred_boxes[pred_lb:pred_rb+1]):
                assert(detection[1] == c)
                
                best_iou = 0.0
                best_gt_idx = -1
                idx = -1
                for gt in true_boxes[true_lb:true_rb+1]:
                    if(gt[0] != detection[0]):
                        continue
                    assert(gt[1] == c)
                    idx += 1
                    iou = intersection_over_union(
                        torch.tensor(detection[3:]),
                        torch.tensor(gt[3:]),
                        box_format=box_format,
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou > iou_threshold:                        
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                        TP[detection_idx] = 1
                        amount_bboxes[detection[0]][best_gt_idx] = 1
                    else:
                        FP[detection_idx] = 1
                else:
                    FP[detection_idx] = 1
                
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_boxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        
        average_precisions.append(torch.trapz(precisions, recalls))
    
    return sum(average_precisions)/len(average_precisions)