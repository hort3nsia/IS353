# Hướng dẫn Hiện thực Đồ án: Multi-relational Link Prediction in Biological Networks with Graph Autoencoders

**Mục tiêu:** Xây dựng mô hình học biểu diễn đồ thị (Graph Representation Learning) để dự đoán các liên kết đa quan hệ (Multi-relational Link Prediction) trong mạng lưới sinh học, cụ thể là tương tác thuốc - protein và protein - protein.

---

## 1. Giới thiệu bài toán (1 điểm)

### 1.1. Vấn đề thực tiễn
Mạng lưới sinh học (Biological Networks) trong thực tế vô cùng phức tạp. Thay vì chỉ có một loại liên kết đơn thuần (Homogeneous), các thực thể sinh học tương tác với nhau qua nhiều loại quan hệ khác nhau (Heterogeneous / Multi-relational). 
* **Ví dụ:** Thuốc A có thể "ức chế" (inhibit) Protein B, nhưng lại "kích hoạt" (activate) Protein C. Hai loại thuốc kết hợp với nhau có thể sinh ra "tác dụng phụ X" (side effect).
* **Hạn chế của phương pháp cũ:** Các thuật toán đồ thị truyền thống hoặc GNN cơ bản (như GCN, GAT) thường thiết kế cho đồ thị đơn quan hệ, dẫn đến việc mất đi thông tin ngữ nghĩa quan trọng từ các loại cạnh (edge types) khác nhau.

### 1.2. Giải pháp Graph Neural Networks (GNN)
Để giải quyết vấn đề này, đồ án sử dụng kiến trúc **Graph Autoencoder (GAE)** hoặc **Variational Graph Autoencoder (VGAE)** kết hợp với **Multi-relational Decoder**:
1.  **Nén thông tin (Encoder):** Sử dụng mạng GNN đa quan hệ để tổng hợp thông tin từ láng giềng và loại liên kết, tạo ra các vector nhúng (embeddings) mang ngữ nghĩa sâu cho từng node (Thuốc, Protein).
2.  **Giải mã (Decoder):** Từ các vector nhúng trong không gian latent, một bộ giải mã đặc thù sẽ tính toán xác suất tồn tại của *một loại liên kết cụ thể* giữa hai node bất kỳ.

---

## 2. Dữ liệu (1 điểm)

### 2.1. Nguồn Dữ liệu
* **Tập dữ liệu:** BioSNAP-Polypharmacy (Stanford BioSNAP) hoặc Zitnik's Multi-relational Dataset.
* **Đặc điểm:** Chứa mạng lưới tương tác Thuốc - Thuốc và Thuốc - Protein với hơn 200 loại quan hệ (relation types) khác nhau, đại diện cho các tương tác và tác dụng phụ đa dạng.

### 2.2. Trực quan hóa dữ liệu (EDA)
Cần hiện thực các script bằng `matplotlib` và `networkx`:
1.  **Phân bố quan hệ (Relation Distribution):** Vẽ biểu đồ Bar Chart thống kê số lượng cạnh (edges) cho từng loại quan hệ (relation type). Nhằm phát hiện xem dữ liệu có bị mất cân bằng (imbalanced) hay không.
2.  **Trực quan hóa đồ thị con (Subgraph Visualization):** Trích xuất một subgraph nhỏ (khoảng 20-50 nodes). Sử dụng `networkx` để vẽ đồ thị, trong đó:
    * Node thuốc: Hình vuông (màu xanh).
    * Node protein: Hình tròn (màu đỏ).
    * Cạnh: Tô màu khác nhau (color-coded) dựa trên loại quan hệ.

---

## 3. Cơ sở lý thuyết & Kiến trúc Model (2 điểm)

### 3.1. Các tài liệu tham khảo cốt lõi
Trong báo cáo cần trích dẫn và tóm tắt ngắn gọn các paper sau:
1.  *Variational Graph Autoencoders for Biological Link Prediction*
2.  *Decagon: Multi-relational Link Prediction in Biomedical Networks*
3.  *Multi-relational Graph Representation Learning for Drug-Gene Interactions*
4.  *Self-Supervised Graph Autoencoders for Prediction of Bio-interactions*
5.  *A Survey on Multi-relational Link Prediction in Knowledge Graphs*

### 3.2. Cấu trúc Mô hình
Mô hình đi theo kiến trúc Autoencoder gồm 2 thành phần chính:

#### A. Encoder (Bộ mã hóa) - R-GCN
Sử dụng **Relational Graph Convolutional Network (R-GCN)**. R-GCN tính toán biểu diễn của node $i$ ở layer $l+1$ bằng cách tổng hợp thông tin từ các node láng giềng $j$, có tính đến loại quan hệ $r$:

$$h_i^{(l+1)} = \sigma \left( \sum_{r \in R} \sum_{j \in N_i^r} \frac{1}{c_{i,r}} W_r^{(l)} h_j^{(l)} + W_0^{(l)} h_i^{(l)} \right)$$

*Trong đó:* $N_i^r$ là tập các node láng giềng của $i$ thông qua quan hệ $r$. $W_r^{(l)}$ là ma trận trọng số riêng cho từng loại quan hệ $r$. Kết quả cuối cùng của Encoder là ma trận nhúng $Z$.

#### B. Decoder (Bộ giải mã) - DistMult
Để đánh giá xác suất có liên kết loại $r$ giữa node $u$ và node $v$, ta dùng hàm tính điểm đa quan hệ **DistMult**. DistMult sử dụng một ma trận đường chéo $W_r$ cho mỗi loại quan hệ:

$$f_r(u, v) = z_u^T W_r z_v$$

Hàm Sigmoid có thể được áp dụng lên $f_r(u,v)$ để chuyển thành xác suất $\in [0, 1]$.

---

## 4. Hiện thực & Đánh giá (4 điểm)

Sử dụng `PyTorch` và `PyTorch Geometric` (PyG) để hiện thực.

### 4.1. Các bước thực hiện chi tiết (Pipeline)

1.  **Data Processing:** * Đọc file raw, chuyển hóa thành định dạng Triplets: `(Subject, Relation, Object)`.
    * Tạo dictionary map ID của node và ID của relation.
    * Chuyển đổi thành object `torch_geometric.data.Data` với thuộc tính `edge_index` (kích thước `[2, num_edges]`) và `edge_type` (kích thước `[num_edges]`).
2.  **Encoder Construction:**
    * Khởi tạo node features ($X$) ban đầu (có thể dùng Embedding layer hoặc Identity matrix nếu không có feature).
    * Xây dựng 2 lớp `RGCNConv` từ PyG. Đầu ra là node embeddings $Z$.
3.  **Decoder Implementation:**
    * Viết class `DistMultDecoder` quản lý tập hợp các ma trận trọng số `W_r` (dùng `torch.nn.Embedding` cho relations).
    * Input: $z_u, z_v, r$. Output: Điểm số của triplet.
4.  **Negative Sampling (Lấy mẫu âm):**
    * Với mỗi cạnh thực (positive) `(u, r, v)`, sinh ra $K$ cạnh giả (negative) `(u, r, v')` bằng cách giữ nguyên subject $u$ và relation $r$, nhưng lấy ngẫu nhiên một node $v'$ không có liên kết thực sự với $u$ qua $r$.
5.  **Training & Loss Function:**
    * **GAE:** Loss = Binary Cross Entropy (BCE) giữa điểm số dự đoán của positive/negative edges và nhãn thật (1/0).
    * **VGAE:** Loss = $\mathcal{L}_{Reconstruction} + \mathcal{L}_{KL}$.
    $$\mathcal{L} = \mathbb{E}_{q(Z|X,A)}[\log p(A|Z)] - \text{KL}[q(Z|X,A) || p(Z)]$$

### 4.2. Cách thức Đánh giá & Báo cáo

#### A. Tối ưu & So sánh (1 điểm)
* **Thực nghiệm:** Train cả 2 mô hình (GAE thông thường sử dụng R-GCN và Variational GAE).
* **So sánh:** Lập bảng so sánh độ đo (AUC, AUPRC) giữa hai model.
* **Biện luận:** Thêm noise vào tập train (xóa ngẫu nhiên 10% cạnh hoặc thêm cạnh giả). Chứng minh VGAE duy trì performance tốt hơn GAE do bản chất học phân bố xác suất thay vì điểm cố định, giúp chống overfitting (robust to noise).

#### B. Trực quan kết quả (1 điểm)
1.  **Biểu đồ ROC-AUC:**
    * Tính điểm ROC-AUC riêng biệt cho từng loại relation.
    * Lọc ra top 5 relations phổ biến nhất (dựa trên EDA ở phần 2) và vẽ đường ROC curve (với 5 màu khác nhau) trên cùng một biểu đồ.
2.  **t-SNE Visualization:**
    * Trích xuất ma trận nhúng $Z$ sau khi train.
    * Dùng thuật toán t-SNE giảm chiều xuống 2D.
    * Vẽ scatter plot: Các node đại diện cho protein. Có thể tô màu protein theo cụm chức năng sinh học của chúng (nếu metadata có cung cấp). Mục tiêu là chứng minh: *Các protein có chức năng tương tự hoặc tương tác nhiều với nhau sẽ co cụm lại thành các cluster trên đồ thị 2D.*

#### C. Diễn giải Mô hình (1 điểm)
* *Case study:* Chọn một triplet cụ thể có điểm số dự đoán cao, ví dụ: `(Thuốc A, Inhibition, Protein B)`.
* *Giải thích cơ chế:* In ra danh sách các láng giềng (neighborhood) bậc 1 và bậc 2 của Thuốc A và Protein B. 
* *Biện luận:* "Mô hình dự đoán Thuốc A ức chế Protein B với xác suất 0.92 vì GNN đã tổng hợp thông tin rằng: Thuốc A có cấu trúc tương tự Thuốc C (láng giềng gần), mà Thuốc C đã được biết là ức chế Protein B mạnh mẽ. Trọng số quan hệ 'Inhibition' trong ma trận của DistMult khớp chặt chẽ với vector latent của hai node này."