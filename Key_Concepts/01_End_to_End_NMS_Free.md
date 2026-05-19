# Suy luận End-to-End Không cần NMS

## 1. Khái niệm NMS (Non-Maximum Suppression) là gì?
Trong các mạng phát hiện đối tượng truyền thống, một đối tượng thường bị phát hiện bởi nhiều "bounding box" (hộp bao quanh) cùng một lúc. Để giải quyết, người ta dùng NMS:
- Lọc các box có cùng lớp (class).
- Giữ lại box có điểm tự tin (confidence score) cao nhất.
- Bỏ đi các box khác nếu mức độ chồng lấn (IoU - Intersection over Union) của nó với box tốt nhất vượt qua một ngưỡng (threshold) nhất định.

**Ví dụ:** Bạn chụp ảnh một con chó. Mạng YOLO cũ có thể dự đoán 5 cái khung đều bao quanh con chó đó. NMS sẽ chạy một vòng lặp để giữ lại khung chuẩn nhất và xóa 4 khung còn lại.

## 2. Vấn đề của NMS
- **Chậm:** NMS chạy trên CPU hoặc yêu cầu kernel tùy chỉnh trên GPU, làm tăng độ trễ (latency).
- **Phụ thuộc tham số:** Cần phải chỉnh tay ngưỡng IoU. Nếu IoU thấp, mô hình có thể xóa nhầm đối tượng đứng sát nhau. Nếu IoU cao, mô hình có thể giữ lại các hộp trùng lặp.

## 3. Giải pháp của YOLO26: End-to-End NMS-Free
YOLO26 **loại bỏ hoàn toàn NMS** trong quá trình suy luận (inference). 
Thay vì dự đoán nhiều hộp cho một đối tượng, mô hình được huấn luyện để **chỉ xuất ra duy nhất một hộp tốt nhất** ngay từ đầu. 

### Cách hoạt động trong Code (Tham chiếu YOLO26)
Trong lúc huấn luyện, YOLO26 dùng **Dual Assignment** (sẽ giải thích kỹ ở phần Head). Nhưng lúc chạy thực tế (inference), nó chỉ dùng nhánh `one2one`.

```python
# Trích xuất từ main/yolo26_modules.py - Class Detect
def postprocess(self, preds):
    boxes, scores = preds.split([4, self.nc], dim=-1)
    # Lấy top-K dự đoán có điểm số cao nhất mà không cần tính toán IoU (NMS)
    index = scores.amax(-1).topk(min(self.max_det, scores.shape[1]))[1].unsqueeze(-1)
    boxes = boxes.gather(1, index.expand(-1, -1, 4))
    scores = scores.gather(1, index.expand(-1, -1, self.nc))
    scores, cls = scores.flatten(1).topk(min(self.max_det, scores.shape[1]))
    
    return torch.cat([boxes.gather(1, (cls // self.nc).unsqueeze(-1).expand(-1, -1, 4)),
                      scores.unsqueeze(-1), (cls % self.nc).unsqueeze(-1).float()], -1)
```

**Tham số liên quan:**
- `max_det` (thường là 300): Số lượng hộp tối đa được giữ lại. Thay vì dùng vòng lặp IoU phức tạp để lọc, YOLO26 chỉ đơn giản là gọi hàm `.topk()` lấy top 300 hộp có điểm số (`scores`) cao nhất toàn ảnh.
