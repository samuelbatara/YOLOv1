import torch 
import torch.nn as nn
from iou import intersection_over_union

class YoloLoss(nn.Module):
    """
    Menghitung nilai loss Yolo v1
    """

    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S: ukuran split (7 di paper)
        B: jumlah bounding box (2 di paper)
        C: jumlah kelas (20 di paper)
        """
        self.S = S
        self.B = B
        self.C = C
        
        """
        lambda_noobj: bobot error untuk cell yang tidak ada objek
        lambda_coord: bobot error untuk kesalahan koordinat (x, y, width, height)
        """
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # Mengubah ukuran prdiksi dari (BATCH_SIZE, S*S*(C + 5*B))
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Menghitung iou dari dua kotak prediksi dengan kotak target
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Mengambil kotak prediksi dengan iou terbesar
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)

        "Loss dari kotak prediksi (x, y, width, height)"
        box_predictions = exists_box * (
            bestbox * predictions[..., 26:30] + (1-bestbox) * predictions[..., 21:25]
        )

        box_targets = exists_box * target[..., 21:25]

        # Menggunakan akar kuadrat dari width dan height
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        "Loss dari sel yang ada objek"
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        "Loss dari sel yang tidak ada objek"
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )
        
        "Loss untuk kesalahan prediksi kelas"
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,),
            torch.flatten(exists_box * target[..., :20], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss  # dua baris pertama di paper
            + object_loss  # baris ketiga di paper
            + self.lambda_noobj * no_object_loss  # baris keempat di paper
            + class_loss  # baris kelima di paper
        )

        return loss