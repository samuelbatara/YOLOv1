# You Only Look One: Unified, Real-Time Object Detection

## Architecture
<figure>
  <img src="images\architecture of YOLOv1.png" alt="loss">
  <figcaption>Gambar 2. Arsitektur YOLOv1 [1].</figcaption>
</figure> <br> <br>
<p><b>Deskripsi: </b></p>
<ul> 
    <li>Input: 448 x 488 x 3</li>
    <li>Output: 7 * 7 * 30</li>
    <li>Ukuran split (S): 7</li>
    <li>Jumlah bounding box: 2</li>
    <li>Jumlah kelas: 20</li>
</ul>
<p>Hasil prediksi diencode sebagai S x S x (B * 5 + C)</p>

## Loss
<figure>
  <img src="images\loss function of YOLOv1.png" alt="loss">
  <figcaption>Gambar 2. Fungsi loss YOLOv1 [1].</figcaption>
</figure> <br><br>
<p><b>Deskripsi: </b></p>
<ul>
    <li><b>Baris pertama dan kedua</b>, loss untuk kesalahan predikdi koordinat (x, y, width, height).</li>
    <li><b>Baris ketiga</b>, loss untuk kesalahan iou score untuk sel yang ada objek.</li>
    <li><b>Baris keempat</b>, loss untuk kesalahan iou score untuk sel yang tidak ada objek.</li>
    <li><b>Baris kelima</b>, loss untuk kesalahan klasifikasi</li>
    <li><b>Lambda_coord</b>, bobot untuk loss koordinat, sebesar 5.</li>
    <li><b>Lambda_noobj</b>, bobot untuk kesalahan iou score untuk sel yang tidak ada objek.</li>
</ul> 
<p>Perhitungan loss menggunakan <b>sum-squared error</b>.</p>

## Refrensi
<p>[1] <a href="https://arxiv.org/abs/1506.02640">https://arxiv.org/abs/1506.02640</a></p>

