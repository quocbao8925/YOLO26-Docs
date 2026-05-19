# Khối C2PSA (CSP with Spatial Attention)

## 1. Hạn chế của CNN truyền thống và Khái niệm Self-Attention
- **CNN (Tích chập):** Một khối Conv 3x3 chỉ có thể "nhìn thấy" 9 điểm ảnh lân cận xung quanh nó. Để nhìn thấy toàn cảnh, mạng phải cực sâu. Nó thiếu tầm nhìn tổng quát (Global Receptive Field).
- **Self-Attention (Tự chú ý):** Kỹ thuật nổi tiếng từ các mô hình Transformer (như ChatGPT, ViT). Nó tính toán "độ liên quan" (attention score) giữa **bất kỳ cặp pixel nào** với nhau dù cho chúng nằm ở hai đầu đối diện của bức ảnh. Ví dụ: Nó có thể hiểu rằng bóng đen dưới mặt nước là hình ảnh phản chiếu của ngọn núi phía trên.

## 2. PSA Block (Position-Sensitive Attention)
Nếu chỉ xài hoàn toàn Transformer cho hình ảnh thì mô hình sẽ cực kỳ nặng nề và chậm chạp. YOLO26 kết hợp cả hai bằng việc tạo ra **C2PSA**.

C2PSA bao bọc khối tự chú ý PSA bên trong một cấu trúc nhánh chéo CSP. Điều này giúp mô hình lấy được tầm nhìn toàn cảnh của Transformer nhưng vẫn giữ được độ nhẹ nhàng, tiết kiệm toán hạng nhờ cấu trúc CSP.

## 3. Code Cấu trúc trong YOLO26

```python
# Trích xuất từ main/yolo26_modules.py
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        # ... Khởi tạo Q, K, V (Query, Key, Value) cơ bản của Transformer ...
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False) # Positional Encoding
    
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        # Sinh ma trận Query, Key, Value
        qkv = self.qkv(x)
        # Tính toán ma trận Attention (Tích vô hướng giữa Query và Key)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        # Biến đổi thành phân phối xác suất
        attn = attn.softmax(dim=-1)
        # Nhân kết quả sự chú ý với Value và khôi phục về kích thước ma trận ảnh
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)

class C2PSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        # Kiến trúc rẽ nhánh CSP
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        # Chuỗi PSA Block
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        # Tách làm 2 nhánh a và b
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        # Chỉ nhánh b đi qua cơ chế Tự chú ý cực nặng
        b = self.m(b)
        # Nối nhánh b và nhánh tắt a lại với nhau, rồi tính chập một lần cuối
        return self.cv2(torch.cat((a, b), 1))
```

## 4. Lý do C2PSA nằm ở cuối Backbone?
Trong kiến trúc, C2PSA là khối **cuối cùng** của Backbone:
```python
self.b10 = C2PSA(ch(1024), ch(1024), n=rep(2))
```
Vì phép toán Tự chú ý tính toán khoảng cách của **tất cả** các pixel, nên kích thước không gian càng to thì độ phức tạp tăng theo cấp số nhân (hàm mũ). Khi tới vị trí `b10`, bản đồ đặc trưng đã được thu nhỏ tới mức tối đa (ví dụ từ ảnh 640x640 xuống chỉ còn lưới 20x20). Tính toán Attention trên lưới nhỏ 20x20 là hoàn hảo, vừa thu được toàn cảnh, vừa giữ được chuẩn thời gian thực (real-time).
