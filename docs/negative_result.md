# Báo cáo kết quả âm: GNN/RL không vượt được heuristic tham lam trên 0/1 Knapsack

*Ngày: 2026-10-23 · Nhánh: `dev` · Phần cứng: RTX 5060 Laptop 8 GB, WSL2 4 CPU*

## Tóm tắt

Chúng tôi huấn luyện một GNN có giám sát và ba phương pháp RL (REINFORCE, DQN,
S2V-DQN) cho bài toán 0/1 knapsack, rồi so sánh với DP, Branch & Bound (B&B),
Greedy theo tỉ lệ v/w và GA. Thí nghiệm dùng hai họ dữ liệu: dữ liệu tương quan yếu
với n từ 10 đến 210 (cross_scale) và Pisinger type 5 "almost strongly correlated"
với n = 100.

Kết luận chính:

1. **Mọi mô hình học được đều hội tụ về chính sách greedy-ratio.** Xác suất mà mô hình
   xuất ra có tương quan hạng Spearman 0.78–0.97 với tỉ lệ v/w. DQN tốt nhất cho
   ra *đúng* lời giải của Greedy. Trên Pisinger, GNN và REINFORCE không rời khỏi
   lời giải greedy trong suốt quá trình huấn luyện.
2. **Phần cải thiện thật đến từ bước tìm kiếm (DP trên top-m), không đến từ GNN.**
   Xếp hạng item bằng tỉ lệ v/w rồi chạy cùng bước DP cho kết quả gần như bằng
   GNN+DP (chênh ≤ 0.0004).
3. **B&B áp đảo.** B&B cho nghiệm tối ưu chính xác. Trên cross_scale, B&B chạy
   0.2–2.2 ms, nhanh hơn GNN (3–6 ms). Trên Pisinger t5, B&B chậm hơn (232 ms)
   nhưng vẫn nhanh hơn DP.
4. Chỉ có một trường hợp mô hình học được vượt Greedy khi đứng một mình:
   REINFORCE trên n nhỏ (0.9937 so với 0.9889). Nhưng ratio+DP đạt 0.9993 với
   chi phí không đáng kể, và B&B giải chính xác trong 0.17 ms.

Đây là một kết quả âm hợp lệ. Trên các phân phối đã thử, 0/1 knapsack một ràng
buộc không để lại khoảng trống nào cho một bộ chấm điểm học được.

## 1. Thiết lập

### Dữ liệu

| Bộ | Sinh bằng | Số instance | n | Ghi chú |
|---|---|---|---|---|
| `cross_scale/train_small` | `Generate_Data.py`, seed 0 | 1001 | 10–50 | dùng để train |
| `cross_scale/val_small` | seed 100 | 201 | 10–50 | |
| `cross_scale/test_small` | seed 200 | 201 | 10–50 | |
| `cross_scale/test_n100` | seed 300 | 201 | 90–110 | chỉ test (tổng quát hoá theo kích thước) |
| `cross_scale/test_n200` | seed 400 | 201 | 190–210 | chỉ test |
| `pisinger_t5/{train,val,test}` | `Generate_Hard.py --type 5` | 1000 / 200 / 200 | 100 | almost strongly correlated |

Trong `Generate_Data.py`, trọng số w ~ U{1..1500} và giá trị v = ⌊w · U(0.8, 1.3)⌋.
Đây là phân phối tương quan yếu. Dung lượng C tăng tuyến tính theo chỉ số
instance, từ gần 0 đến gần Σw. Nghiệm tối ưu được tính bằng ILP (PuLP).

Với Pisinger, `genhard` seed theo id instance. Vì vậy chúng tôi sinh một pool
1400 instance bằng một lệnh duy nhất rồi chia theo chỉ số file. Cách này tránh
việc train/val/test bị trùng instance.

### Solver

- **DP**: quy hoạch động, cho nghiệm tối ưu và dùng làm mốc (ratio = 1).
- **B&B**: branch & bound với cận LP, timeout 5 s (không instance nào chạm timeout).
- **Greedy**: sắp item theo v/w giảm dần, nhét item nào còn vừa.
- **GA**: thuật toán di truyền.
- **GNN**: GNN có giám sát (hidden 256, 4 layer, đồ thị kNN k = 16), giải mã
  greedy theo xác suất.
- **GNN+DP**: DP trên top-m item theo xác suất GNN, sau đó lấp dung lượng còn lại
  theo xác suất (m = 40 trong `Run_all_evaluations.py`).
- **REINFORCE**: khởi tạo từ GNN có giám sát rồi fine-tune bằng policy gradient.
- **DQN**, **S2V-DQN**: dựng nghiệm tuần tự, mỗi bước chọn một item.

Đặc trưng nút của GNN gồm `w_norm`, `v_norm` và **`ratio_norm` = (v/w)/max(v/w)**
(`src/gnn/Knapsack_GNN/Graph_builder.py:133`). Như vậy mô hình được cho sẵn khoá
sắp xếp của Greedy trong đầu vào.

Chỉ số đánh giá là **ratio** = giá trị lời giải / giá trị tối ưu DP, lấy trung bình
trên tập test. Tất cả solver đều cho lời giải khả thi trên 100% instance.

## 2. Kết quả

### 2.1 cross_scale (train trên n 10–50)

| Solver | small (n 10–50) | n100 | n200 | thời gian ở n200 |
|---|---|---|---|---|
| DP | 1.0000 | 1.0000 | 1.0000 | 2246 ms |
| **B&B** | **1.0000** | **1.0000** | **1.0000** | **2.2 ms** |
| GNN+DP | 0.9995 | 0.9992 | 0.9995 | 24.9 ms |
| GA | 0.9983 | 0.9980 | 0.9959 | 53.9 ms |
| REINFORCE | 0.9937 | 0.9984 | 0.9991 | 4.8 ms |
| **Greedy** | 0.9889 | 0.9986 | 0.9996 | **0.04 ms** |
| DQN (best ckpt) | 0.9889 | 0.9986 | 0.9996 | 57.6 ms |
| GNN | 0.9850 | 0.9975 | 0.9989 | 6.3 ms |
| S2V-DQN | 0.9385 | 0.8963 | 0.8957 | 169 ms |

Nhận xét:

- Ở n100 và n200, Greedy đứng trên mọi mô hình học được khi các mô hình này
  đứng một mình.
- DQN cho ra *đúng* số của Greedy trên cả ba tập.
- Khoảng cách từ Greedy đến tối ưu giảm nhanh khi n tăng (1.1% → 0.14% → 0.04%).

### 2.2 Pisinger type 5, n = 100 (train và test cùng phân phối)

| Solver | ratio | thời gian |
|---|---|---|
| DP | 1.0000 | 386 ms |
| **B&B** | **1.0000** | 232 ms |
| GA | 0.9977 | 44.9 ms |
| GNN+DP | 0.9890 | 11.1 ms |
| GNN | 0.9864 | 5.4 ms |
| REINFORCE (baseline none) | 0.9864 | 5.6 ms |
| REINFORCE (baseline polyak) | 0.9864 | — |
| **Greedy** | 0.9863 | **0.03 ms** |
| DQN (best ckpt) | 0.9863 | 28.0 ms |
| S2V-DQN | 0.9172 | 78.8 ms |

Trên dữ liệu khó này, Greedy kém tối ưu 1.4%. Dù vậy, GNN chỉ hơn Greedy
0.0001, còn GA (không học gì) hơn mọi phương pháp học được. Số của REINFORCE
polyak lấy từ `greedy_equivalence.py`.

## 3. Phân tích: mô hình học được chính là Greedy

Script `src/scripts/greedy_equivalence.py` so sánh trên từng instance. Với mỗi
mô hình, script đo:

- **same**: tỉ lệ instance mà lời giải greedy-theo-xác-suất trùng hệt lời giải
  greedy-theo-v/w;
- **ρ**: hệ số Spearman giữa xác suất mô hình và v/w;
- **ratio+DP30** so với **model+DP30**: cùng bước DP trên top-30 item, chỉ khác
  thứ tự xếp hạng (v/w hay xác suất mô hình).

| Tập | Greedy | ratio+DP30 | GNN | GNN+DP30 | GNN same | GNN ρ | RF | RF+DP30 | RF same | RF ρ |
|---|---|---|---|---|---|---|---|---|---|---|
| small | 0.9889 | 0.9993 | 0.9850 | 0.9995 | 0.59 | 0.81 | 0.9936 | 0.9996 | 0.30 | 0.78 |
| n100 | 0.9986 | 0.9991 | 0.9975 | 0.9992 | 0.40 | 0.87 | 0.9984 | 0.9990 | 0.08 | 0.85 |
| n200 | 0.9996 | 0.9997 | 0.9989 | 0.9995 | 0.25 | 0.89 | 0.9991 | 0.9992 | 0.01 | 0.87 |
| Pisinger t5 | 0.9863 | 0.9889 | 0.9864 | 0.9890 | 0.74 | 0.97 | 0.9864 | 0.9890 | 0.74 | 0.97 |

(RF trên Pisinger: số của cả hai biến thể none và polyak trùng nhau đến 4 chữ số;
ρ = 0.971.)

Diễn giải:

- **ratio+DP30 ≈ GNN+DP30 trên mọi tập.** Chênh lệch lớn nhất là 0.0004, và ở
  n200 ratio+DP còn tốt hơn. Toàn bộ mức tăng của GNN+DP so với GNN đến từ bước DP.
  Bộ xếp hạng nào gần đúng thứ tự v/w cũng cho cùng kết quả.
- **ρ cao (0.78–0.97)** cho thấy mô hình về cơ bản đang tái tạo `ratio_norm`, một
  đặc trưng có sẵn trong đầu vào.
- Tỉ lệ "same" thấp ở n lớn (ví dụ GNN chỉ 0.25 ở n200) *không* có nghĩa mô hình học
  được điều gì khác. Lời giải lệch khỏi Greedy ở vài item biên và đều tệ hơn Greedy.
  Khi mô hình khác Greedy, nó khác theo chiều xấu đi.

### Vì sao không có chỗ để cải thiện

Gọi OPT là giá trị tối ưu. Khi sắp item theo v/w, cận LP cho:

OPT ≤ (tổng giá trị tiền tố greedy) + v_critical.

Từ đó:

Greedy ≥ OPT − max_i v_i.

Sai số tương đối của Greedy vì vậy bị chặn bởi v_max / OPT. Đại lượng này giảm
theo 1/(số item trong nghiệm), tức xấp xỉ O(1/n) khi C/Σw cố định. Hệ quả:

- Mọi bộ chấm điểm chỉ có thể kiếm thêm tối đa một item ở biên, và phần lợi đó
  co lại khi n tăng.
- Vị trí của item biên phụ thuộc tổ hợp vào C. Một bộ chấm điểm *độc lập theo
  từng item* (đầu ra GNN) rất khó nắm bắt điều này. Bước DP/tìm kiếm thì nắm bắt
  được, với chi phí nhỏ.

Pisinger type 5 được thiết kế để mọi v/w gần bằng nhau, nên v/w mang ít thông tin.
Dù vậy, GNN vẫn không tìm được tín hiệu nào tốt hơn: ρ = 0.97 và lời giải trùng
Greedy trên 74% instance.

## 4. Quan sát khi huấn luyện RL

- **DQN (cross_scale).** Checkpoint tốt nhất xuất hiện ở bước 5000 (val 0.9846), và
  đó chính là chính sách greedy. Sau đó mô hình *suy giảm*, đến khoảng 0.85–0.87 ở
  bước 60k. Nếu dùng nhầm `dqn.pt` (checkpoint cuối) thay vì `dqn_best.pt`, kết
  quả chỉ còn 0.889 ở n100. Trên Pisinger, DQN tốt nhất trùng Greedy (0.9863).
- **REINFORCE (Pisinger).** Mô hình được warm-start từ GNN. Cả hai biến thể
  baseline (none, polyak) đều có best ở epoch 1, và `val_avg_value` đứng yên ở
  29793.6 suốt 40 epoch. Nguyên nhân:
  - Policy đã gần như tất định (entropy ≈ 0.13).
  - Độ lệch chuẩn của advantage (≈ 200) chỉ bằng ~0.6% reward (≈ 32 000).
  - Với lr 5e-5, gradient không đủ lớn để đảo thứ tự xác suất.
  - Thay đổi baseline không có tác dụng.
- **GNN có giám sát (Pisinger).** Val ratio đứng yên ở 0.9871 suốt 60 epoch. So
  với Greedy trên tập val: 0 thắng, 61 thua, 139 hoà.
- **S2V-DQN.** Đây là phương pháp yếu nhất (0.89–0.94), tệ hơn Greedy rất xa. Val
  vẫn còn tăng ở bước 60k, nên mô hình có thể chưa hội tụ (xem mục 6). Tuy vậy,
  ngay cả khi hội tụ, mức trần hợp lý của nó vẫn chỉ là Greedy.

## 5. B&B áp đảo

| | small | n100 | n200 | Pisinger t5 (n100) |
|---|---|---|---|---|
| B&B (chính xác) | 0.17 ms | 0.8 ms | 2.2 ms | 232 ms |
| DP (chính xác) | 59 ms | 562 ms | 2246 ms | 386 ms |
| GNN (xấp xỉ) | 4.8 ms | 3.4 ms | 6.3 ms | 5.4 ms |
| GNN+DP (xấp xỉ) | 6.7 ms | 13.2 ms | 24.9 ms | 11.1 ms |

- **Trên cross_scale:** B&B với cận LP vừa *tối ưu chính xác* vừa *nhanh hơn* GNN.
  Lợi thế "GNN nhanh hơn DP" (≈ 5 ms so với 2.2 s) chỉ đúng khi so với DP. Đó là
  một baseline yếu: độ phức tạp của DP là O(nC), tăng theo dung lượng.
- **Trên Pisinger t5:** đây là nơi duy nhất GNN nhanh hơn B&B (5 ms so với 232 ms).
  Nhưng GNN chỉ đạt 0.9864 = Greedy, mà Greedy chạy trong 0.03 ms.

## 6. Hạn chế và mối đe doạ tính hợp lệ

- **Phân phối dữ liệu.** Mới thử dữ liệu tương quan yếu và Pisinger type 5. Chưa
  thử strongly correlated, inverse strongly correlated, subset-sum (type 6),
  spanner hay circle (các type khó nhất của Pisinger).
- **Kích thước.** Train ở n ≤ 50 (cross_scale) và n = 100 (Pisinger). Test tối đa
  n = 210. Chưa thử n ≥ 1000 hoặc C rất lớn, là vùng mà DP và B&B có thể chậm.
- **Siêu tham số.**
  - Phân tích DP chỉ thử top_m = 30 (script) và 40 (pipeline).
  - Các mô hình RL dùng cấu hình mặc định trong `configs/`, không tuning.
  - S2V-DQN có thể chưa hội tụ.
  - REINFORCE có thể cần lr lớn hơn hoặc thêm entropy bonus để thoát khỏi warm start.
- **Thiết kế đặc trưng.** `ratio_norm` là đặc trưng đầu vào. Kết quả "mô hình
  bằng Greedy" một phần là do thiết kế này. Bỏ đặc trưng này *có thể* buộc mô hình
  học cách khác, nhưng không có lý do để kỳ vọng nó vượt v/w (mục 3).
- **Một lần chạy.** Mỗi cấu hình chỉ chạy một seed, không có khoảng tin cậy.
  Tuy vậy, các chênh lệch giữa Greedy và mô hình nằm ở chữ số thứ 4 và nhất quán
  trên mọi tập.

## 7. Tái lập

```bash
# Dữ liệu cross_scale
G=".venv/bin/python src/scripts/Generate_Data.py --quiet"
$G 1001 10 50   -p data/cross_scale/train_small -s 0
$G 201  10 50   -p data/cross_scale/val_small   -s 100
$G 201  10 50   -p data/cross_scale/test_small  -s 200
$G 201  90 110  -p data/cross_scale/test_n100   -s 300
$G 201  190 210 -p data/cross_scale/test_n200   -s 400

# Dữ liệu Pisinger type 5: sinh một pool, rồi chia theo chỉ số file thành
# train/val/test = 1000/200/200 (các id phải khác nhau giữa các split)
.venv/bin/python src/scripts/Generate_Hard.py --type 5 --n_items 100 \
    --num_instances 1400 --series 2000 --seed 0 --out_dir data/pisinger_t5/pool

# Huấn luyện cross_scale (với Pisinger: trỏ các đường dẫn train/val/test sang
# data/pisinger_t5/*; REINFORCE thêm --hidden_dim 256 --num_layers 4 --pretrained <gnn_best.pt>)
.venv/bin/python src/gnn/Run_train.py --config configs/gnn_supervised.yaml \
    --generated_dir data/cross_scale/train_small --val_dir data/cross_scale/val_small \
    --test_dir data/cross_scale/test_small --save_path results/GNN/gnn_best.pt
.venv/bin/python src/rl/REINFORCE/Train_gnn_reinforce.py --config configs/reinforce.yaml
.venv/bin/python src/rl/DQN/train_dqn.py --config configs/dqn.yaml --device cuda
.venv/bin/python src/rl/S2V_DQN/train_s2v_dqn.py --config configs/s2v_dqn.yaml --device cuda

# Bảng 2.1 (lặp lại cho t in small n100 n200)
.venv/bin/python src/scripts/Run_all_evaluations.py \
    --dataset_dir data/cross_scale/test_$t --results_dir results/compare_$t \
    --model_path results/GNN/gnn_best.pt --dqn_model results/DQN/dqn_best.pt \
    --s2v_model results/S2V_DQN/s2v_dqn_full/s2v_dqn_full_best.pt --s2v_graph_type full \
    --reinforce_model results/GNN_REINFORCE/gnn_reinforce_knn/gnn_reinforce_knn.pt \
    --bb_timeout 5

# Bảng mục 3
.venv/bin/python src/scripts/greedy_equivalence.py \
    --test_dirs data/cross_scale/test_small data/cross_scale/test_n100 data/cross_scale/test_n200 \
    --models GNN=results/GNN/gnn_best.pt \
             RF=results/GNN_REINFORCE/gnn_reinforce_knn/gnn_reinforce_knn.pt \
    --top_m 30
```

## 8. Hướng tiếp theo

Hướng đi hợp lý là những bài toán mà Greedy *không* gần tối ưu và giải chính
xác thực sự đắt:

1. **Multi-dimensional knapsack (MKP).** Khi có nhiều ràng buộc, không còn một tỉ
  lệ v/w duy nhất. Greedy có thể kém tối ưu nhiều phần trăm, và B&B/ILP tăng
  theo hàm mũ. Đây là nơi GNN trên đồ thị item–ràng buộc có lý do để tồn tại.
2. **Dung lượng rất lớn / n rất lớn.** Ở đây DP O(nC) không khả thi. Nhưng lưu ý:
  với 0/1 knapsack một ràng buộc, B&B và Greedy vẫn mạnh, nên hướng này chỉ có
  ý nghĩa khi kết hợp với (1).
3. **ML-guided B&B.** Thay vì thay thế solver, dùng GNN để chọn biến rẽ nhánh hoặc
  đặt thứ tự node trong B&B/ILP (kiểu Gasse et al., 2019). Khi đó mô hình được đo
  bằng thời gian giải *chính xác*, không bằng chất lượng nghiệm xấp xỉ.
4. Nếu vẫn tiếp tục với 0/1 knapsack, mọi đề xuất nên được so với **ratio+DP** và
  **B&B** chứ không chỉ với Greedy hay DP. Đây là hai baseline mà báo cáo này cho
  thấy là then chốt.
