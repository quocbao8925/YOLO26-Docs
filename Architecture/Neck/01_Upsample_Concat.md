# Khối Upsample và Concat (Cấu trúc Cổ - Neck)

## 1. Vai trò của phần Cổ (Neck) - FPN & PAN
- **Mạng xương sống (Backbone)** trích xuất đặc trưng và liên tục làm nhỏ bức ảnh (downsampling). Ở cuối mạng, ta có 1 bức tranh tuy mờ nhạt (độ phân giải thấp 20x20) nhưng nắm rõ bản chất **"Cái gì nằm ở đâu"** (Ngữ cảnh cao). Ở đầu mạng (lưới 80x80), ta có độ nét cực tốt nhưng mô hình chỉ hiểu những đoạn mã **"Cạnh góc nhỏ"**.
- Nếu dự đoán ngay, mô hình sẽ mù màu với đồ vật lớn (nếu dùng lưới nhỏ) hoặc bỏ lỡ vật thể bé tí (nếu dùng lưới to).
- **Phần Neck (Cổ)** sinh ra để trộn lẫn 2 cái đó. Nó sẽ đưa cái ngữ cảnh từ đáy mạng dâng lên, và ghép thông tin nét từ đầu mạng đi xuống. Kỹ thuật này gọi là kết hợp **FPN (Feature Pyramid Network)** (từ dưới lên) và **PAN (Path Aggregation Network)** (từ trên xuống).

## 2. Các khối cấu tạo

- **Upsample (Tăng độ phân giải):** Dùng thuật toán Nội suy gần nhất (Nearest Neighbor). Nó chỉ đơn giản nhân bản giá trị pixel kế bên để phóng to ma trận gấp 2 lần mà không cần học các tham số mới.
- **Concat (Nối ma trận):** Hàm của PyTorch (`torch.cat`), chồng 2 tấm lưới (feature map) lên nhau theo trục chiều sâu.

## 3. Khám phá Code tạo nên phần Neck YOLO26

Phần Cổ sẽ tận dụng 3 lớp cuối của Backbone là `b4` (P3/chi tiết cao), `b6` (P4/chi tiết vừa), và `b10` (P5/chi tiết thấp - ngữ cảnh cao).

```python
# Trích xuất từ main/yolo26_modules.py
# ---- Bắt đầu phần Neck ----

# 1. Đi từ dưới lên (FPN: Phóng to đặc trưng ngữ cảnh)
self.up1 = nn.Upsample(scale_factor=2, mode="nearest") # Phóng to b10 gấp 2 lần (vd 20x20 -> 40x40)
self.h13 = C3k2(...)  # Nối b10 đã phóng to với b6 (Concat), rồi xử lý trộn bằng C3k2

self.up2 = nn.Upsample(scale_factor=2, mode="nearest") # Tiếp tục phóng to h13 gấp 2 (vd 40x40 -> 80x80)
self.h16 = C3k2(...)  # Nối h13 đã phóng to với b4 -> Tạo thành lưới P3 (Chuyên phát hiện đồ vật tí hon)

# 2. Đi từ trên xuống (PAN: Thu nhỏ đặc trưng hình thái sắc nét dội xuống lại)
self.h17 = Conv(..., 3, 2) # Tích chập Stride=2 để thu nhỏ lưới P3 một nửa (80x80 -> 40x40)
self.h19 = C3k2(...) # Nối P3 thu nhỏ với đoạn h13 (40x40) ở trên -> Tạo thành lưới P4 (Chuyên đồ vật kích thước vừa)

self.h20 = Conv(..., 3, 2) # Tiếp tục thu nhỏ P4 (40x40 -> 20x20)
self.h22 = C3k2(..., attn=True) # Nối P4 vừa thu nhỏ vào cục gốc b10 -> Tạo thành lưới P5 (Chuyên đồ vật khổng lồ)
```

**Sự liên kết hoàn hảo:**
Bằng cách liên tục `Upsample` (kéo lên) rồi lại `Conv Stride=2` (kéo xuống), YOLO26 đảm bảo:
- **Lưới P3** có thông tin của vật cực nhỏ nhưng hiểu được toàn cảnh bức tranh (vì có dòng chảy từ b10 lên).
- **Lưới P5** bắt vật siêu bự nhưng vẫn sắc cạnh, rõ biên giới hình dáng (vì có dòng chảy từ P3 dội xuống lại).
Đó là bí quyết khiến YOLO là vua trong việc phát hiện đối tượng đa kích thước.
