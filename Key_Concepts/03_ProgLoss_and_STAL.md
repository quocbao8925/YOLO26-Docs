# ProgLoss và STAL

## 1. Vấn đề Phát hiện Đối tượng Nhỏ
Các đối tượng nhỏ (như người ở rất xa, chim trên trời, biển số xe) chiếm lượng pixel cực kỳ ít trong ảnh (ví dụ 8x8 pixel). Khi đi qua nhiều lớp Convolution, đặc trưng của chúng dễ bị mờ và "biến mất". Thêm vào đó, trong lúc huấn luyện tính Loss (Hàm mất mát), mô hình thường ưu tiên tối ưu để nhận diện đúng các vật thể to, dễ nhìn hơn.

## 2. STAL (Small-Target-Aware Label Assignment)
STAL (Gán nhãn nhận thức mục tiêu nhỏ) là một kỹ thuật ép buộc mô hình chú ý vào các vật thể bé.

**Cơ chế:**
- Bình thường, khi gán một ground-truth (đáp án) cho các anchor (điểm neo) trên feature map, nếu vật thể quá nhỏ, nó có thể chỉ khớp với 1 hoặc thậm chí 0 anchor nào.
- STAL cưỡng ép: **Mọi đối tượng nhỏ hơn 8x8 pixel phải được gán ít nhất cho 4 anchors.**

**Ví dụ thực tế:**
Ảnh có 1 chiếc xe tải to đùng và 1 con chim bé xíu. Không có STAL, xe tải có thể được phân bổ cho 20 anchors (mô hình học xe tải rất kỹ), con chim chỉ được gán 1 anchor (mô hình lơ là con chim). Có STAL, con chim sẽ bị ép gán cho 4 anchors, khiến độ quan trọng của nó trong Loss tăng lên gấp 4 lần, buộc mô hình phải học cách nhận ra nó.

## 3. ProgLoss (Progressive Loss Balancing)
Trong quá trình huấn luyện, mô hình học 2 nhiệm vụ song song:
1. **Classification Loss:** Phân loại (Con chó hay con mèo?)
2. **Bounding Box Loss:** Hồi quy khung (Vẽ khung to nhỏ ra sao?)

**Cơ chế của ProgLoss:**
- Hàm mất mát này **thay đổi trọng số linh hoạt** theo từng giai đoạn huấn luyện (epoch).
- Giai đoạn đầu (Epoch thấp): Ưu tiên mô hình học vị trí (Box Loss) để tìm ra "vật thể nằm ở đâu" trước.
- Giai đoạn sau (Epoch cao): Khi mô hình đã vẽ khung khá chuẩn, ProgLoss tăng trọng số để tinh chỉnh khả năng nhận diện phân loại (Class Loss).

Sự kết hợp giữa ProgLoss và STAL là chìa khóa giúp YOLO26 đặc biệt xuất sắc và có độ chính xác (mAP) cực cao trong việc phát hiện vật thể nhỏ bị che khuất mà không đánh đổi sự ổn định tổng thể.
