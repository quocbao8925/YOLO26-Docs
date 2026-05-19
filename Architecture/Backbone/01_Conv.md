# Khối Tích Chập (Conv Block)

## 1. Khái niệm Convolution
Khối Conv (Tích chập) là nền tảng cốt lõi của mọi mạng nơ-ron hình ảnh (CNN). Nhiệm vụ của nó là dùng một ma trận nhỏ (gọi là Kernel / Filter) trượt dọc theo bức ảnh để trích xuất các đặc trưng: từ các nét cơ bản (cạnh thẳng, góc, màu sắc) cho tới các hình dạng phức tạp (mắt, mũi, bánh xe).

Trong cấu trúc của YOLO26, khối `Conv` không chỉ là một phép toán tích chập toán học, mà nó là một chuỗi 3 thao tác kết hợp: **Conv2d + BatchNorm2d + Kích hoạt (Activation)** (thường gọi là khối CBS - Conv/Batch/Silu).

## 2. Code và Tham số trong YOLO26

```python
# Trích xuất từ main/yolo26_modules.py
class Conv(nn.Module):
    default_act = nn.SiLU() # Hàm kích hoạt mặc định là SiLU (Swish)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        # c1: Số kênh (channel) đầu vào
        # c2: Số kênh (channel) đầu ra (quyết định số lượng đặc trưng)
        # k: Kích thước kernel (vd: k=3 tức là ma trận 3x3)
        # s: Stride (bước nhảy)
        # p: Padding (chèn viền xung quanh để không làm nhỏ ảnh nếu không muốn)
        # g: Groups (sử dụng khi cần tạo Depthwise Convolution)
        # d: Dilation (Tích chập giãn - mở rộng vùng nhìn nhưng giữ nguyên số lượng biến)
        
        # 1. Phép tích chập
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # 2. Chuẩn hóa Batch (Tránh hiện tượng gradient biến mất, tăng tốc học)
        self.bn = nn.BatchNorm2d(c2)
        # 3. Hàm kích hoạt phi tuyến tính (Cho phép mô hình học các hàm phức tạp)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
```

## 3. Ví dụ sử dụng trong Backbone

Lớp đầu tiên của YOLO26 khi nhận ảnh đầu vào:
```python
self.b0 = Conv(3, ch(64), 3, 2)          # P1/2
```
- **Phân tích:** Nhận ảnh đầu vào có `c1=3` (3 kênh màu RGB). Số kênh đầu ra `c2=ch(64)`. Kích thước filter `k=3` (3x3). Đặc biệt có bước nhảy `s=2` (Stride = 2).
- **Tác dụng của Stride=2:** Mỗi lần trượt, kernel sẽ nhảy cách 2 pixel. Điều này làm cho kích thước không gian của bức ảnh (Width, Height) bị **giảm đi một nửa** (downsampling). Đổi lại, thông tin được "dồn" vào chiều sâu (số kênh tăng lên 64), giúp rút gọn khối lượng tính toán cho các lớp mạng sâu hơn.
