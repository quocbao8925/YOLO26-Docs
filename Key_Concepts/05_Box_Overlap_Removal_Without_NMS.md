# Giải pháp Loại bỏ Box Overlap Không dùng NMS của YOLO26

Trong các phiên bản YOLO cũ (như YOLOv8), mô hình thường tạo ra rất nhiều bounding box (hộp giới hạn) chồng chéo lên cùng một vật thể. Sau đó, nó phải dùng thuật toán NMS (Non-Maximum Suppression) để lọc và giữ lại hộp tốt nhất. Quá trình này làm chậm tốc độ suy luận (inference) và gây khó khăn khi triển khai trên các thiết bị yếu.

Để loại bỏ NMS mà không làm giảm độ chính xác, YOLO26 (lấy cảm hứng từ YOLOv10) đã sử dụng chiến lược huấn luyện gán nhãn kép (dual assignment) với hai đầu ra (head) là **one-to-many (o2m)** và **one-to-one (o2o)**. 

Dưới đây là cách hai head này phối hợp để thay thế hoàn toàn NMS:

## 1. Vai trò của từng Head trong giai đoạn Huấn luyện (Training)

Trong lúc training, YOLO26 sử dụng đồng thời cả hai head này để tận dụng ưu điểm của từng phương pháp:

- **One-to-Many Head (o2m - Một chạm Nhiều):** Head này cho phép liên kết nhiều hộp dự đoán với cùng một vật thể thực tế (ground-truth). Điều này giúp cung cấp một lượng lớn tín hiệu giám sát (dense supervision) phong phú, giúp các phần mạn sườn (backbone) và cổ (neck) của mạng nơ-ron học hỏi các đặc trưng của hình ảnh một cách toàn diện và ổn định trong giai đoạn đầu.
  
- **One-to-One Head (o2o - Một chạm Một):** Ngược lại, head này ép mô hình chỉ được phép liên kết duy nhất một hộp dự đoán cho mỗi vật thể. Quá trình này rèn luyện cho mạng nơ-ron khả năng tự nhận biết và trực tiếp xuất ra một bounding box duy nhất và chính xác nhất cho vật thể, thay vì xuất ra nhiều hộp dư thừa. Đây chính là "chìa khóa" để loại bỏ NMS.

## 2. Sự can thiệp của ProgLoss (Progressive Loss Balancing)

Nếu chỉ chạy song song 2 head, mô hình sẽ bị "bối rối". YOLO26 giải quyết việc này bằng hàm mất mát **ProgLoss**.

- Ở những epoch đầu, ProgLoss sẽ dồn trọng số vào o2m head để mô hình học các đặc trưng cơ bản và tăng tỷ lệ phát hiện (recall).
- Càng về cuối quá trình training, trọng số sẽ dần chuyển sang o2o head. Điều này ép mô hình dần dần chuyển sang trạng thái chỉ dự đoán một hộp duy nhất, chuẩn bị sẵn sàng cho quá trình suy luận thực tế.

## 3. Điều gì xảy ra ở giai đoạn Suy luận (Inference)?

Đây là lúc phép màu NMS-free thực sự hoạt động:

- **Cắt bỏ o2m head:** Khi mô hình đã train xong và mang đi sử dụng thực tế (inference), YOLO26 sẽ vứt bỏ hoàn toàn o2m head. Mô hình lúc này chỉ sử dụng duy nhất o2o head để đưa ra dự đoán.
- **Không cần so sánh độ lệch (IoU):** Vì o2o head đã được huấn luyện bài bản để không sinh ra các hộp trùng lặp, YOLO26 không cần phải tính toán độ giao nhau (IoU) giữa các hộp để lọc như NMS nữa.
- **Sử dụng Top-K:** Thay vào đó, mô hình chỉ đơn giản là xếp hạng các dự đoán dựa trên điểm số tự tin (confidence scores) và chọn ra K kết quả cao nhất (Top-K selection).

**Tóm lại:** 
YOLO26 dùng o2m head như một "bánh xe tập" để dạy mô hình học sâu các đặc trưng, và dùng o2o head để dạy mô hình tính quyết đoán (chỉ chọn 1 hộp). Khi mang đi sử dụng, nó tháo "bánh xe tập" o2m ra, chỉ giữ lại o2o. Nhờ vậy, YOLO26 có thể dự đoán trực tiếp (end-to-end) ra kết quả cuối cùng, bỏ qua bước lọc NMS rườm rà, giúp tăng tốc độ suy luận trên CPU lên tới 43%.
