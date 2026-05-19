# Chiến lược Gán Nhãn Kép (Dual Assignment Strategy)

## 1. Nguồn gốc Vấn đề NMS
Để có thể loại bỏ thuật toán NMS (Non-Maximum Suppression) trong quá trình Inference, mô hình bắt buộc phải tự "giác ngộ" và học được cách dự đoán chính xác **một hộp (box) duy nhất** cho mỗi đồ vật.

Tuy nhiên, nếu khi đưa ảnh vào huấn luyện, ta chỉ phạt mô hình theo dạng "1 đấu 1" (chỉ 1 hộp dự đoán trúng 1 đáp án), mô hình học rất lâu, mông lung và tính toán cập nhật Gradient cực kỳ kém hiệu quả vì lượng dữ liệu đáp ứng quá thưa thớt (sparse). 

## 2. Giải pháp Gán Nhãn Kép (Dual Assignment)
Kế thừa từ tinh hoa của YOLOv10, YOLO26 sử dụng chiến lược chạy 2 luồng dự đoán (Head) song song lúc huấn luyện:

1. **One-to-Many Head (Nhiều chọi 1):** Một vật thể cho phép nhiều lưới / nhiều điểm neo dự đoán trúng cùng một lúc. Giúp cung cấp luồng Gradient phong phú mạnh mẽ, dạy mô hình trích xuất đặc trưng chung tốt nhất.
2. **One-to-One Head (1 chọi 1):** Mỗi vật thể lưới chỉ cho phép duy nhất 1 hộp dự đoán chuẩn xác nhất được tính Loss dương, còn lại phạt thẳng tay. Ép mô hình tự học cách triệt tiêu các hộp xung quanh.

## 3. Hoạt động thực tiễn trong Code YOLO26

Ở phần Head `Detect`, hai mạng neural riêng biệt sẽ được sinh ra (NẾU `end2end=True`):

```python
# Trích xuất từ main/yolo26_modules.py - Class Detect
class Detect(nn.Module):
    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        # ...
        # Khởi tạo nhánh One-to-Many gốc tiêu chuẩn của kiến trúc YOLO
        self.cv2 = nn.ModuleList(...) # Nhánh dự đoán vị trí Hộp (Box head)
        self.cv3 = nn.ModuleList(...) # Nhánh dự đoán Loại (Class head)
        
        # TÍNH NĂNG END-TO-END MỚI CỦA YOLO26
        if end2end:
            # Tạo bản sao y hệt song song, làm nhánh One-to-One
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        # 1. Cho tín hiệu x đi qua nhánh One-to-Many
        preds = self.forward_head(x, **self.one2many)
        
        # 2. Cho tín hiệu x đi qua nhánh One-to-One
        if self.end2end:
            # Dùng detach() ngắt gradient để nhánh One-to-One không gây nhiễu, không cập nhật trọng số Backbone
            x_detach = [xi.detach() for xi in x] 
            one2one = self.forward_head(x_detach, **self.one2one)
            
            # Gói gọn cả hai kết quả
            preds = {"one2many": preds, "one2one": one2one}
            
        # NẾU ĐANG HUẤN LUYỆN (Training): Trả về cả hai kết quả để ném vào hàm Loss
        if self.training:
            return preds
            
        # NẾU ĐANG CHẠY THỰC TẾ (Inference): Vứt hẳn One-to-Many, chỉ chạy One-to-One
        y = self._inference(preds["one2one"] if self.end2end else preds)
        
        # Xử lý Top-K (không NMS)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y
```

**Đỉnh cao của việc tối ưu hóa:**
- Lúc ở phòng tập (Training Mode), mô hình vác trên lưng 2 cái tạ (2 khối Head) để tối đa hoá khả năng học.
- Lúc ra chiến trường (Inference Mode), toàn bộ khối lượng tham số nặng nề của cụm `cv2`, `cv3` (one2many) hoàn toàn bị bỏ qua, không tính toán. Khối Head lúc này nhẹ hều và tốc độ đạt chuẩn Real-time!
