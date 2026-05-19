# YOLO26: Trình Phát Hiện Đối Tượng Thời Gian Thực Thế Hệ Mới

*Tổng quan Kiến trúc và Các Cải tiến Chính*

---

## Mục lục

1. [Các Khái niệm Chính](#cac-khai-niem-chinh)
   - [Suy luận End-to-End Không cần NMS](#suy-luan-end-to-end-khong-can-nms)
   - [Loại bỏ Distribution Focal Loss (DFL)](#loai-bo-distribution-focal-loss-dfl)
   - [ProgLoss và STAL](#progloss-va-stal)
   - [Bộ tối ưu hóa MuSGD](#bo-toi-uu-hoa-musgd)
2. [Kiến trúc YOLO26](#kien-truc-yolo26)
   - [Backbone (Mạng xương sống)](#backbone-mang-xuong-song)
   - [Neck (Phần cổ)](#neck-phan-co)
   - [Detection Head (Đầu phát hiện)](#detection-head-dau-phat-hien)
3. [Huấn luyện YOLO26](#huan-luyen-yolo26)
   - [Chiến lược Gán Nhãn Kép (Dual Assignment Strategy)](#chien-luoc-gan-nhan-kep-dual-assignment-strategy)
   - [Điều chỉnh Hàm mất mát (Loss)](#dieu-chinh-ham-mat-mat-loss)
4. [Xử lý Dự đoán](#xu-ly-du-doan)
   - [Lựa chọn Top-K Dựa trên Điểm số](#lua-chon-top-k-dua-tren-diem-so)
5. [Hiệu suất & Đánh giá (Benchmarking)](#hieu-suat--danh-gia-benchmarking)

---

## Các Khái niệm Chính

### Suy luận End-to-End Không cần NMS

Các kiến trúc YOLO trước đây phụ thuộc nhiều vào Non-Maximum Suppression (NMS) như một bước hậu xử lý để loại bỏ các dự đoán trùng lặp. NMS làm tăng độ trễ (latency) và đòi hỏi phải tinh chỉnh siêu tham số (hyperparameter). YOLO26 giới thiệu một bộ dự đoán end-to-end gốc giúp loại bỏ hoàn toàn NMS. Điều này dẫn đến thời gian suy luận trên CPU nhanh hơn tới 43% và đơn giản hóa việc triển khai trên các thiết bị biên (edge devices).

### Loại bỏ Distribution Focal Loss (DFL)

DFL được sử dụng trong các phiên bản trước (như YOLOv8) để dự đoán phân phối xác suất cho tọa độ bounding box. YOLO26 loại bỏ DFL, quay trở lại phương pháp hồi quy trực tiếp các tọa độ bounding box rõ ràng. Sự đơn giản hóa này giúp giảm khối lượng tính toán và đảm bảo việc xuất mô hình (export) sang các định dạng như ONNX, TensorRT, và CoreML sạch sẽ và gọn gàng hơn.

### ProgLoss và STAL

- **ProgLoss (Progressive Loss Balancing):** Tự động điều chỉnh trọng số của các thành phần hàm mất mát khác nhau trong suốt quá trình huấn luyện để ngăn chặn tình trạng overfitting vào các lớp (classes) chiếm đa số.
- **STAL (Small-Target-Aware Label Assignment):** Giải quyết trực tiếp thách thức trong việc phát hiện các đối tượng nhỏ. Phương pháp này đảm bảo tối thiểu bốn anchor cho các đối tượng nhỏ hơn 8x8 pixel, đảm bảo chúng luôn đóng góp vào hàm mất mát trong quá trình huấn luyện.

### Bộ tối ưu hóa MuSGD

YOLO26 giới thiệu **MuSGD**, một bộ tối ưu hóa lai kết hợp giữa Stochastic Gradient Descent (SGD) với các bản cập nhật kiểu Muon (thường được sử dụng trong các Mô hình Ngôn ngữ Lớn - LLMs). Điều này cho phép mô hình hội tụ nhanh hơn, mượt mà hơn và ổn định hơn.

---

## Kiến trúc YOLO26

Kiến trúc của YOLO26 được đặc trưng bởi ba thành phần chính: Backbone, Neck, và Head. Nó được thiết kế để có khả năng mở rộng cao (scalable) bằng cách sử dụng các tham số `depth_multiple` và `width_multiple` để tạo thành các biến thể từ Nano (n) đến Extra-Large (x).

### Backbone (Mạng xương sống)

Backbone xử lý hình ảnh đầu vào để trích xuất các đặc trưng ở nhiều tỷ lệ khác nhau:

| Khối (Block) | Loại | Mô tả |
|---|---|---|
| **b0** | Conv | Conv 3x3, Stride 2 (P1/2) |
| **b1** | Conv | Conv 3x3, Stride 2 (P2/4) |
| **b2** | C3k2 | CSP Bottleneck với 2 tích chập (convolutions) |
| **b3** | Conv | Conv 3x3, Stride 2 (P3/8) |
| **b4** | C3k2 | Trích xuất đặc trưng sâu hơn |
| **b5** | Conv | Conv 3x3, Stride 2 (P4/16) |
| **b6** | C3k2 | Sử dụng `c3k=True` cho các phép toán với kernel lớn hơn |
| **b7** | Conv | Conv 3x3, Stride 2 (P5/32) |
| **b8** | C3k2 | Trích xuất đặc trưng mức độ cao (high-level) |
| **b9** | SPPF | Spatial Pyramid Pooling Fast, **được cập nhật thêm một kết nối shortcut** |
| **b10** | C2PSA | Giới thiệu cơ chế tự chú ý (self-attention) với PSA Block để nắm bắt ngữ cảnh toàn cục |

### Neck (Phần cổ)

Neck tổng hợp các đặc trưng từ backbone thông qua việc upsample (tăng độ phân giải) và nối (concatenation) để tạo thành Feature Pyramid Network (FPN) và Path Aggregation Network (PAN).

| Khối (Block) | Loại | Mô tả |
|---|---|---|
| **up1** | Upsample | Upsampling 2x bằng Nearest Neighbor cho khối `b10` |
| **h13** | Concat + C3k2 | Nối `up1(b10)` với `b6` |
| **up2** | Upsample | Upsampling 2x bằng Nearest Neighbor cho khối `h13` |
| **h16** | Concat + C3k2 | Nối `up2(h13)` với `b4` → **Xuất ra feature map P3** |
| **h17** | Conv | Conv 3x3, Stride 2 để downsampling |
| **h19** | Concat + C3k2 | Nối `h17` với `h13` → **Xuất ra feature map P4** |
| **h20** | Conv | Conv 3x3, Stride 2 để downsampling |
| **h22** | Concat + C3k2 | Nối `h20` với `b10`. Tích hợp `attn=True` (PSA Block) → **Xuất ra feature map P5** |

### Detection Head (Đầu phát hiện)

Head của YOLO26 hoạt động trên các feature map P3, P4 và P5.

Trong quá trình khởi tạo, hai head song song riêng biệt sẽ được tạo ra nếu `end2end=True`:
1. **One-to-Many Head (`one2many`)**: Được sử dụng độc quyền trong quá trình huấn luyện để cung cấp các gradient phong phú.
2. **One-to-One Head (`one2one`)**: Được sử dụng để suy luận end-to-end mà không cần NMS.

Mỗi head bao gồm:
- **Box Branch (Nhánh hộp):** Chuỗi các Conv dự đoán `4 * reg_max` giá trị (với `reg_max=1` do đã loại bỏ DFL).
- **Class Branch (Nhánh lớp):** Chuỗi các Depthwise Convs (DWConv) dự đoán điểm số cho `nc` (số lượng lớp).

---

## Huấn luyện YOLO26

### Chiến lược Gán Nhãn Kép (Dual Assignment Strategy)

Lấy cảm hứng từ YOLOv10, YOLO26 sử dụng chiến lược gán nhãn kép trong quá trình huấn luyện. Nó sử dụng cả phương pháp gán one-to-many truyền thống (nơi nhiều anchor có thể khớp với một ground truth) và phương pháp gán one-to-one.
- Nhánh one-to-many cung cấp sự giám sát dày đặc (dense supervision).
- Nhánh one-to-one học cách dự đoán bounding box đơn lẻ tốt nhất cho mỗi đối tượng.

### Điều chỉnh Hàm mất mát (Loss)

Quá trình huấn luyện phụ thuộc nhiều vào cơ chế **ProgLoss** để cân bằng động giữa hàm mất mát phân loại (classification) và hồi quy (regression), trong khi **STAL** buộc mạng lưới phải phạt nặng các mục tiêu nhỏ bị bỏ sót, giúp tăng đáng kể mAP cho các đối tượng nhỏ.

---

## Xử lý Dự đoán

### Lựa chọn Top-K Dựa trên Điểm số

Trong quá trình suy luận, chỉ có head `one2one` được sử dụng. Vì mạng đã được huấn luyện để chỉ xuất ra một box tốt nhất cho mỗi đối tượng, bước hậu xử lý (post-processing) được đơn giản hóa đi rất nhiều:

1. Trích xuất tọa độ bounding box và điểm phân loại được dự đoán.
2. Chọn điểm số cao nhất cho mỗi bounding box.
3. Chỉ giữ lại **Top-K** dự đoán hoàn toàn dựa trên điểm phân loại, mà không cần tính toán Intersection over Union (IoU) giữa các box.
4. Xuất trực tiếp các phát hiện cuối cùng.

Việc dự đoán trực tiếp này làm giảm đáng kể sự biến động về độ trễ (latency variance) giữa các hình ảnh khác nhau.

---

## Hiệu suất & Đánh giá (Benchmarking)

YOLO26 được tối ưu hóa cho thiết bị biên (edge), mang lại sự cân bằng tuyệt vời giữa tốc độ và độ chính xác:

- **Tốc độ:** Biến thể Nano (`YOLO26n`) đạt tốc độ suy luận 38.9 ms trên CPU và 1.7 ms trên GPU NVIDIA T4 sử dụng TensorRT FP16.
- **Độ chính xác:** `YOLO26n` đạt 40.9 mAP(50-95), và mở rộng lên đến 57.5 mAP đối với phiên bản `YOLO26x`.
- **Đa nhiệm (Tasks):** Ngoài phát hiện đối tượng tiêu chuẩn, kiến trúc hợp nhất của YOLO26 hỗ trợ nguyên bản cho Phân vùng thực thể (Instance Segmentation), Ước lượng Điểm chính/Tư thế (Pose/Keypoints Estimation), Phát hiện Định hướng (Oriented Detection - OBB) và Phân loại (Classification) với thay đổi kiến trúc cực kỳ tối thiểu.

---

*Được tạo ra nhằm mục đích phân tích dựa trên các Bài báo về YOLO26 và Mã nguồn được triển khai.*
