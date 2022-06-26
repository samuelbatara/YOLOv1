def is_smaller(XX, YY):
    """
    Menentukan berdasarkan priotitas:
        1. XX[1] < YY[1]: kelas yang lebih kecil
        2. XX[2] > YY[2]: iou yang lebih besar
        
    Parameters:
        XX, YY (list): [train_idx, class_pred, preb_score, x, y, width, height]
        
    Returns:
        bool: True jika XX dan False jika tidak XX
    """
    if XX[1] < YY[1]:
        return True 
    if XX[1] == YY[1] and XX[2] > YY[2]:
        return True
    return False

def merge(boxes, left, middle, right):
    kiri = [xx for xx in boxes[left:middle+1]]
    kanan = [xx for xx in boxes[middle+1:right+1]] 
    i, j, idx = 0, 0, left

    while(left + i <= middle and middle + 1 + j <= right):
        if(is_smaller(kiri[i], kanan[j])):
            boxes[idx] = kiri[i]
            i += 1
        else:
            boxes[idx] = kanan[j]
            j += 1
        idx += 1
    
    while(left + i <= middle):
        boxes[idx] = kiri[i]
        i += 1
        idx += 1
    while(middle + 1 + j <= right):
        boxes[idx] = kanan[j]
        j += 1
        idx += 1

def merge_sort(boxes, left, right):
    """
    Mengurutkan kotak prediksi dengan teknik Merge Sort
    
    Parameters:
        boxes (list): [[train_idx, class_pred, prob_score, x, y, width, height]]
        left (int): batas kiri
        right (int): batas kanan

    Returns:
        list: bboxes setelah terurut
    """
    
    if(left < right):
        middle = left + (right-left)//2
        merge_sort(boxes, left, middle) 
        merge_sort(boxes, middle+1, right) 
        merge(boxes, left, middle, right)
    
    return boxes

def left_bound(boxes, c, idx=1):
    """
    Menentukan index lower bound dari kelas c
    
    Parameters:
        boxes (list): [[train_idx, class_pred, prob_score, x, y, width, height]]
        c (int): kelas
        idx (int): index pembanding
        
    Returns:
        int: index lower bound
    """
    
    if(len(boxes) == 0):
        return 0
    
    left, right = 0, len(boxes)-1
    while(left < right):
        middle = left + (right-left)//2 
        if(boxes[middle][idx] >= c):
            right = middle 
        else:
            left = middle + 1
    
    assert left == right, "Method left_bound terjadi bug"
    return left