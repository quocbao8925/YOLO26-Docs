# Loại bỏ Distribution Focal Loss (DFL)

## 1. DFL (Distribution Focal Loss) là gì?
Trong các phiên bản YOLO trước (như YOLOv8), thay vì dự đoán trực tiếp tọa độ (x, y, w, h) của bounding box, mạng lưới dự đoán một **phân phối xác suất** của các tọa độ.
Nghĩa là thay vì nói "Cạnh trái của hộp nằm ở pixel số 10", nó sẽ nói "Có 10% khả năng ở pixel 9, 80% khả năng ở pixel 10, 10% khả năng ở pixel 11".

**Ưu điểm của DFL:** Giúp mô hình bắt được sự không chắc chắn (ví dụ cạnh của đối tượng bị mờ).
**Nhược điểm:**
- Tốn tính toán.
- Rất khó tối ưu và chuyển đổi (export) sang các định dạng biên như TensorRT hay CoreML vì các phép toán softmax và tính tích phân phức tạp.

## 2. Giải pháp của YOLO26
YOLO26 **loại bỏ DFL** và quay trở lại phương pháp hồi quy trực tiếp 4 giá trị tọa độ (offset).

### So sánh Code
**Có DFL (YOLOv8):**
```python
class DFL(nn.Module):
    # DFL chuyển đổi mảng (batch, 4*reg_max, anchors) thành (batch, 4, anchors)
    def forward(self, x):
        b, _, a = x.shape
        # Phải dùng Softmax và phép nhân ma trận liên tục rất nặng nề
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
```

**Không có DFL (YOLO26):**
Trong YOLO26, tham số `reg_max` được thiết lập cứng về `1` (thay vì 16 như YOLOv8). Do đó, `4 * reg_max` chỉ đơn giản là 4 (tương ứng với dx, dy, dw, dh).
```python
# Trích xuất từ yolo26_modules.py
# Khởi tạo mô hình
self.detect = Detect(nc=nc, reg_max=1, end2end=True, ch=det_ch)

# Bên trong Detect:
self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
# Với reg_max = 1 (YOLO26), self.dfl chỉ là nn.Identity() -> trả về đúng giá trị đầu vào, bỏ qua hoàn toàn block DFL.
```

**Ví dụ thực tế:**
Thay vì tính toán phân phối phức tạp với vector độ dài 16 cho mỗi cạnh (trái, trên, phải, dưới), YOLO26 nhả trực tiếp 4 con số cụ thể, giúp các vi điều khiển (như Jetson Nano, Raspberry Pi) chạy mượt mà và tiết kiệm bộ nhớ RAM hơn rất nhiều.
