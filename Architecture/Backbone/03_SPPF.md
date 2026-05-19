# Khối SPPF (Spatial Pyramid Pooling Fast)

## 1. Khái niệm SPP (Spatial Pyramid Pooling)
Khi đưa ảnh vào các mạng CNN thế hệ cũ, hình ảnh thường bị bóp méo (resize cứng) về một kích thước vuông chuẩn (vd 224x224) làm mất đi tỷ lệ thật. Kỹ thuật SPP giải quyết vấn đề này bằng cách dùng nhiều cửa sổ Pooling (kéo gộp đặc trưng) ở các kích thước khác nhau ghép lại, tạo ra một vector đầu ra có độ dài cố định bất chấp kích cỡ ảnh đầu vào.

**SPPF (Fast SPP)** là phiên bản tốc độ cao: Thay vì dùng 3 cửa sổ Pooling cỡ lớn song song nhau (ví dụ kernel 5x5, 9x9, 13x13), tác giả thiết kế SPPF nối chuỗi 3 cửa sổ nhỏ 5x5 liên tiếp nhau. Kết quả toán học là tương đương, nhưng tốc độ tính toán trên phần cứng GPU tăng lên rất nhiều lần.

## 2. Cải tiến độc quyền của YOLO26 trên SPPF
Ở YOLO26, tác giả cấu trúc lại SPPF bằng cách **thêm vào một kết nối lối tắt (shortcut connection)** ở cuối khối.

**Tại sao phải thêm Shortcut?**
SPPF thường nằm ở các tầng sâu nhất của mạng (cuối Backbone). Ở đây tín hiệu dễ bị "nghẽn" khi tính đạo hàm (Gradient Vanishing). Shortcut cho phép luồng đạo hàm chảy mượt hơn và bảo toàn được thông tin gốc mà không bị qua quá nhiều bước pooling làm biến dạng.

## 3. Code của SPPF trong YOLO26

```python
# Trích xuất từ main/yolo26_modules.py
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5, n=3, shortcut=False):
        super().__init__()
        c_ = c1 // 2
        # Giảm số kênh xuống 1 nửa trước khi xử lý
        self.cv1 = Conv(c1, c_, 1, 1)
        # Nâng kênh lại ở đầu ra
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        # Tạo hàm kéo gộp MaxPooling 2D với kernel cố định là k (thường là 5)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        
        # ĐÂY LÀ ĐIỂM MỚI: Chỉ tạo shortcut nếu số kênh vào = số kênh ra và cờ được bật
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = [self.cv1(x)]
        # Chạy chuỗi Max pooling 5x5 liên tiếp nhau n lần (thường n=3)
        y.extend(self.m(y[-1]) for _ in range(self.n))
        # Nối tất cả các kết quả dọc theo trục chiều sâu channel và đưa qua Conv
        y = self.cv2(torch.cat(y, 1))
        
        # NẾU tính năng add (shortcut) được kích hoạt, cộng giá trị x ban đầu vào y
        return y + x if self.add else y
```

**Ví dụ sử dụng trong Backbone:**
```python
self.b9 = SPPF(ch(1024), ch(1024), k=5, n=3, shortcut=True)
```
- Đầu vào nhận tín hiệu siêu sâu với 1024 channels.
- `shortcut=True`: Tính năng shortcut được bật giúp tránh nghẽn cổ chai tại khối thứ 9 (b9) cực kỳ quan trọng này.
