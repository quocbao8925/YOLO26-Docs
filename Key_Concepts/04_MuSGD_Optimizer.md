# Bộ Tối Ưu Hóa MuSGD (MuSGD Optimizer)

## 1. Optimizer (Bộ tối ưu hóa) là gì?
Khi huấn luyện mô hình AI, Optimizer là thuật toán có nhiệm vụ điều chỉnh cập nhật các trọng số (weights) của mạng neural sau mỗi batch để giảm thiểu sai số (Loss). Các optimizer phổ biến hiện nay bao gồm Adam, AdamW, và SGD.
- **SGD (Stochastic Gradient Descent):** Rất tốt trong việc giúp mô hình tổng quát hóa (generalization - học bản chất chứ không học vẹt), nhưng tốc độ hội tụ khá chậm và dễ mắc kẹt ở các điểm cực tiểu cục bộ.
- **Adam/AdamW:** Học cực nhanh nhưng đôi khi lại cho kết quả kiểm thử (test) kém hơn SGD một chút do khả năng khái quát kém hơn.

## 2. MuSGD là gì?
YOLO26 giới thiệu **MuSGD**, là sự kết hợp lai giữa kỹ thuật **Muon** và **SGD**.
- **Muon** là một kỹ thuật tối ưu hóa mới nổi, vốn được sinh ra dành cho các Mô hình Ngôn ngữ Lớn (LLMs - như GPT, Claude). Nó sử dụng thông tin độ cong (curvature) hoặc ma trận xấp xỉ Hessian để đưa ra các bước cập nhật trọng số chính xác và ổn định hơn ở các chiều không gian có độ dốc cao.
- Việc mang kỹ thuật từ lĩnh vực Xử lý ngôn ngữ tự nhiên (NLP) áp dụng sang Thị giác máy tính (CV) là một nước đi đột phá của YOLO26.

## 3. Lợi ích của MuSGD đối với YOLO26
- **Tốc độ hội tụ:** Giúp mô hình học nhanh hơn, đạt được mAP cao với ít epoch hơn so với việc dùng SGD truyền thống.
- **Tính ổn định:** Tránh hiện tượng Loss bị nhảy vọt (spike) đột ngột ở các epoch giữa hoặc cuối - một hiện tượng rất hay gặp khi huấn luyện YOLOv8/v11. Đường cong học tập (learning curve) trở nên mượt mà.
- **Khái quát cao:** Vẫn duy trì được sức mạnh cốt lõi của SGD, giúp mô hình ra ngoài môi trường thực tế dự đoán mạnh mẽ không kém gì lúc huấn luyện.

*(Lưu ý: Logic thuật toán của optimizer nằm ở tầng Training Engine / Trainer của PyTorch chứ không nằm trong kiến trúc module của mạng nơ-ron tĩnh (nn.Module).)*
