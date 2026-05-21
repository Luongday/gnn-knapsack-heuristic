# GNN for 0/1 Knapsack — Benchmark nhiều solver

Dự án nghiên cứu giải bài toán **0/1 Knapsack** bằng **Graph Neural Network (GNN)**, và so sánh với các phương pháp cổ điển cùng reinforcement learning.

---

## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Yêu cầu hệ thống](#2-yêu-cầu-hệ-thống)
3. [Cài đặt](#3-cài-đặt)
4. [Cấu trúc project](#4-cấu-trúc-project)
5. [Danh sách solver](#5-danh-sách-solver)
6. [Pipeline sinh dữ liệu](#6-pipeline-sinh-dữ-liệu)
7. [Huấn luyện mô hình neural](#7-huấn-luyện-mô-hình-neural)
8. [Đánh giá từng solver](#8-đánh-giá-từng-solver)
9. [Pipeline tự động](#9-pipeline-tự-động)
10. [Vẽ biểu đồ](#10-vẽ-biểu-đồ)
11. [Thí nghiệm nâng cao](#11-thí-nghiệm-nâng-cao)
12. [Streamlit Dashboard](#12-streamlit-dashboard)
13. [Google Colab](#13-google-colab)
14. [Troubleshooting](#14-troubleshooting)
15. [Ghi chú kỹ thuật](#15-ghi-chú-kỹ-thuật)

---

## 1. Tổng quan

### 1.1 Bài toán

**0/1 Knapsack**: cho $n$ item, mỗi item có trọng lượng $w_i$ và giá trị $v_i$, cho dung lượng $C$ của ba lô. Tìm tập con các item để tối đa hoá tổng giá trị sao cho tổng trọng lượng không vượt $C$:

$$\max \sum_{i=1}^{n} v_i x_i \quad \text{s.t.} \quad \sum_{i=1}^{n} w_i x_i \leq C, \quad x_i \in \{0, 1\}$$

### 1.2 Mục tiêu

So sánh nhiều phương pháp giải trên cùng một tập dữ liệu, đánh giá trade-off giữa **chất lượng lời giải** (approximation ratio vs DP) và **thời gian chạy** (inference time).

### 1.3 Metric đánh giá

- **Approximation ratio** (vs DP optimal): `ratio = V_solver / V_DP`
- **Inference time** (ms/instance)
- **Feasibility rate**: tỷ lệ lời giải hợp lệ (không vi phạm capacity)
- **Head-to-head**: số instance mà solver A thắng solver B

**Lưu ý**: DP là ground truth. Mọi solver khác được đánh giá bằng ratio so với DP (DP luôn = 1.0).

---

## 2. Yêu cầu hệ thống

### 2.1 Phần cứng

- **CPU**: 4 cores trở lên (project được thiết kế chạy được trên CPU-only)
- **RAM**: tối thiểu 8 GB; khuyến nghị 16 GB cho dataset lớn (n lên tới 200)
- **Disk**: khoảng 2 GB cho dataset + checkpoint + results
- **GPU** (tùy chọn): CUDA-enabled, giúp tăng tốc training GNN/RL đáng kể

### 2.2 Hệ điều hành

Đã test trên:
- Windows 10/11
- Ubuntu 20.04/22.04

### 2.3 Python

- Python **3.10** hoặc mới hơn

### 2.4 Thư viện chính

Xem `Requirements.txt` để cài đặt chính xác. Các dependency quan trọng:

| Package | Mục đích |
|---|---|
| `torch` | Framework deep learning |
| `torch-geometric` | GNN layers (GINConv, SAGEConv, global_pool) |
| `numpy` | Xử lý mảng |
| `pulp` | ILP solver dùng để sinh dữ liệu |
| `matplotlib`, `seaborn` | Vẽ biểu đồ |
| `pyyaml` | Đọc file cấu hình YAML |
| `streamlit` | Dashboard trực quan |

---

## 3. Cài đặt

### 3.1 Clone hoặc tải project

```bash
git clone <repository-url>
cd GNNForKnapSack
```

### 3.2 Tạo môi trường ảo

**Windows (CMD):**

```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3.3 Cài dependencies

```bash
pip install --upgrade pip
pip install -r Requirements.txt
```

> **Lưu ý torch-geometric**: nếu `pip install torch-geometric` báo lỗi, cài bằng wheel chính thức theo phiên bản PyTorch và CUDA của bạn. Xem: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

### 3.4 Verify cài đặt

```bash
pytest src/scripts/test_solvers.py
```

---

## 4. Cấu trúc project

```text
GNNForKnapSack/
│
├── configs/                         # File cấu hình YAML
│   ├── gnn_supervised.yaml
│   ├── dqn.yaml
│   ├── s2v_dqn.yaml
│   └── reinforce.yaml
│
├── src/                             # Mã nguồn chính
│   ├── core/                        # Tiện ích dùng chung
│   │   ├── config_loader.py         # Load YAML config + resolve_device()
│   │   ├── decode_utils.py          # Greedy / DP-subset / beam decode
│   │   ├── instance_loader.py       # Đọc file .npz
│   │   └── rl_training_logger.py    # Logger cho RL training
│   │
│   ├── gnn/                         # Pipeline GNN có giám sát
│   │   ├── Run_train.py             # Entry point huấn luyện GNN
│   │   ├── Evaluate_GNN.py          # Đánh giá GNN
│   │   ├── Evaluate_GNN_DP.py       # Đánh giá GNN với DP decode
│   │   ├── Evaluate_CallBack.py     # Callback đánh giá trong lúc train
│   │   └── Knapsack_GNN/
│   │       ├── model.py             # Kiến trúc GNN (GINConv + GlobalCtx)
│   │       ├── dataset.py           # Dataset loader
│   │       ├── Graph_builder.py     # Xây dựng đồ thị KNN/Conflict/...
│   │       ├── Train_eval.py        # Vòng lặp train/val
│   │       ├── Dp.py                # DP solver nội bộ
│   │       └── io_excel.py          # Xuất kết quả Excel
│   │
│   ├── rl/                          # Học tăng cường
│   │   ├── DQN/
│   │   │   ├── train_dqn.py         # Huấn luyện DQN
│   │   │   ├── Evaluate_DQN.py      # Đánh giá DQN
│   │   │   ├── dqn_env.py           # Môi trường Knapsack cho DQN
│   │   │   ├── dqn_model.py         # Kiến trúc Q-network (MLP)
│   │   │   ├── dqn_replay.py        # Replay buffer
│   │   │   └── dqn_config.py        # Cấu hình mặc định
│   │   ├── S2V_DQN/
│   │   │   ├── train_s2v_dqn.py     # Huấn luyện S2V-DQN
│   │   │   ├── Evaluate_S2V_DQN.py  # Đánh giá S2V-DQN
│   │   │   ├── s2v_env.py           # Môi trường + graph state
│   │   │   ├── s2v_model.py         # Kiến trúc GNN Q-network
│   │   │   └── s2v_replay.py        # Replay buffer
│   │   └── REINFORCE/
│   │       ├── Train_gnn_reinforce.py  # Huấn luyện REINFORCE
│   │       └── Evaluate_reinforce.py   # Đánh giá REINFORCE
│   │
│   ├── solvers/                     # Baseline cổ điển
│   │   ├── DP/dp_baseline_eval.py   # Dynamic Programming (exact)
│   │   ├── Greedy/greedy_baseline_eval.py  # Greedy (v/w ratio)
│   │   ├── GA/ga_baseline_eval.py   # Genetic Algorithm
│   │   └── BB/bb_baseline_eval.py   # Branch-and-Bound
│   │
│   └── scripts/                     # Công cụ sinh dữ liệu & phân tích
│       ├── Generate_Data.py         # Sinh dữ liệu uncorrelated
│       ├── Generate_Hard.py         # Sinh Pisinger hard instances
│       ├── Check_Data.py            # Kiểm tra tính hợp lệ của data
│       ├── Run_all_evaluations.py   # Pipeline tự động toàn bộ
│       ├── Merge_results.py         # Merge CSV kết quả
│       ├── cross_scale_eval.py      # Đánh giá cross-scale/distribution
│       ├── Benchmark_Hard.py        # Benchmark Pisinger hard sets
│       ├── analyze_results.py       # Phân tích thống kê
│       ├── plot_results.py          # Vẽ biểu đồ so sánh
│       ├── plot_ablation.py         # Vẽ biểu đồ ablation
│       └── test_solvers.py          # Unit tests
│
├── data/                            # Dữ liệu (tự động sinh)
│   ├── knapsack_ilp/
│   │   ├── train/                   # ~1001 instance .npz
│   │   ├── val/                     # ~201 instance .npz
│   │   └── test/                    # ~201 instance .npz
│   ├── cross_scale/                 # Cross-scale benchmark instances
│   │   ├── train_small/             # Train: 10-50 items
│   │   ├── val_small/               # Val: 10-50 items
│   │   ├── test_small/              # Test: 10-50 items
│   │   ├── test_n100/               # Test generalization: ~100 items
│   │   └── test_n200/               # Test generalization: ~200 items
│   └── pisinger/                    # Pisinger hard instances (tùy chọn)
│
├── results/                         # Kết quả (tự động sinh)
│   ├── GNN/                         # gnn_best.pt, gnn_eval_results.csv
│   ├── DQN/                         # dqn_best.pt
│   ├── S2V_DQN/                     # s2v_dqn_best.pt
│   ├── GNN_REINFORCE/               # gnn_reinforce.pt
│   └── compare/                     # Kết quả merge, biểu đồ
│
├── plots/                           # Biểu đồ xuất ra
├── app.py                           # Streamlit dashboard
├── GNNForKnapSack_Colab.ipynb       # Notebook cho Google Colab
├── Requirements.txt
└── README.md
```

---

## 5. Danh sách solver

### 5.1 Nhóm cổ điển

| Solver | File | Đặc điểm |
|---|---|---|
| DP | `src/solvers/DP/dp_baseline_eval.py` | Exact — ground truth |
| Greedy | `src/solvers/Greedy/greedy_baseline_eval.py` | Sắp xếp theo v/w ratio |
| GA | `src/solvers/GA/ga_baseline_eval.py` | Genetic Algorithm |
| BB | `src/solvers/BB/bb_baseline_eval.py` | Branch-and-Bound |

### 5.2 Nhóm neural

| Solver | File huấn luyện | File đánh giá | Ghi chú |
|---|---|---|---|
| GNN supervised | `src/gnn/Run_train.py` | `src/gnn/Evaluate_GNN.py` | Học từ nhãn DP |
| GNN + REINFORCE | `src/rl/REINFORCE/Train_gnn_reinforce.py` | `src/rl/REINFORCE/Evaluate_reinforce.py` | Policy gradient |
| DQN | `src/rl/DQN/train_dqn.py` | `src/rl/DQN/Evaluate_DQN.py` | Deep Q-learning (vector state) |
| S2V-DQN | `src/rl/S2V_DQN/train_s2v_dqn.py` | `src/rl/S2V_DQN/Evaluate_S2V_DQN.py` | Structure2Vec + DQN |

---

## 6. Pipeline sinh dữ liệu

Tất cả lệnh chạy từ **thư mục gốc** của project (`GNNForKnapSack/`).

### 6.1 Sinh dữ liệu uncorrelated

```bash
# Train: 1001 instances
python src/scripts/Generate_Data.py 1001 10 200 -p data/knapsack_ilp/train -s 0

# Val: 201 instances
python src/scripts/Generate_Data.py 201 10 200 -p data/knapsack_ilp/val -s 100

# Test: 201 instances
python src/scripts/Generate_Data.py 201 10 200 -p data/knapsack_ilp/test -s 200
```

### 6.2 Sinh Pisinger hard instances

```bash
# Type 3 (strongly correlated) – khó cho Greedy
python src/scripts/Generate_Hard.py \
    --type 3 --n_items 100 --num_instances 1000 \
    --out_dir data/pisinger/type_03/train
```

### 6.3 Kiểm tra dữ liệu

```bash
python src/scripts/Check_Data.py data/knapsack_ilp/train
```

---

## 7. Huấn luyện mô hình neural

Tất cả lệnh chạy từ **thư mục gốc** của project. Mọi script hỗ trợ flag `--device` để chọn thiết bị:

```
--device auto   # tự động dùng GPU nếu có, CPU nếu không (mặc định)
--device cuda   # bắt buộc dùng GPU
--device cpu    # bắt buộc dùng CPU
```

### 7.1 GNN supervised

```bash
# Dùng file YAML (khuyến nghị)
python src/gnn/Run_train.py --config configs/gnn_supervised.yaml

# Ghi đè tham số
python src/gnn/Run_train.py --config configs/gnn_supervised.yaml \
    --epochs 200 --graph_type conflict_static --device cuda
```

Các loại đồ thị hỗ trợ (`--graph_type`): `knn` (mặc định), `conflict_static`, `random`, `full`

### 7.2 S2V-DQN

```bash
python src/rl/S2V_DQN/train_s2v_dqn.py --config configs/s2v_dqn.yaml --device cuda
```

### 7.3 DQN

```bash
python src/rl/DQN/train_dqn.py --config configs/dqn.yaml --device cuda
```

### 7.4 REINFORCE

```bash
python src/rl/REINFORCE/Train_gnn_reinforce.py --config configs/reinforce.yaml --device cuda
```

---

## 8. Đánh giá từng solver

Tất cả lệnh chạy từ **thư mục gốc** của project.

```bash
# DP (chạy đầu tiên — là ground truth)
python src/solvers/DP/dp_baseline_eval.py \
    --dataset_dir data/knapsack_ilp/test \
    --out_csv results/compare/DP/dp_results.csv

# Greedy
python src/solvers/Greedy/greedy_baseline_eval.py \
    --dataset_dir data/knapsack_ilp/test \
    --out_csv results/compare/Greedy/greedy_eval_results.csv

# GNN
python src/gnn/Evaluate_GNN.py \
    --dataset_dir data/knapsack_ilp/test \
    --model_path results/GNN/gnn_best.pt \
    --graph_type knn \
    --out_csv results/GNN/gnn_eval_results.csv \
    --device auto

# S2V-DQN
python src/rl/S2V_DQN/Evaluate_S2V_DQN.py \
    --dataset_dir data/knapsack_ilp/test \
    --model_path results/S2V_DQN/s2v_dqn_best.pt \
    --out_csv results/compare/S2V_DQN/s2v_dqn_eval_results.csv \
    --device auto

# DQN
python src/rl/DQN/Evaluate_DQN.py \
    --dataset_dir data/knapsack_ilp/test \
    --model_path results/DQN/dqn_best.pt \
    --out_csv results/compare/DQN/dqn_on_small.csv \
    --device auto

# REINFORCE
python src/rl/REINFORCE/Evaluate_reinforce.py \
    --dataset_dir data/knapsack_ilp/test \
    --model_path results/GNN_REINFORCE/gnn_reinforce.pt \
    --out_csv results/compare/GNN_REINFORCE/reinforce_eval_results.csv \
    --device auto
```

Sau khi có các CSV, merge kết quả:

```bash
python src/scripts/Merge_results.py --out_dir results/compare
```

---

## 9. Pipeline tự động

`Run_all_evaluations.py` chạy tất cả solver rồi merge tự động.

```bash
python src/scripts/Run_all_evaluations.py \
    --dataset_dir data/knapsack_ilp/test \
    --model_path results/GNN/gnn_best.pt \
    --dqn_model results/DQN/dqn_best.pt \
    --s2v_model results/S2V_DQN/s2v_dqn_best.pt \
    --reinforce_model results/GNN_REINFORCE/gnn_reinforce.pt
```

Tuỳ chọn hữu ích:

```bash
# Bỏ qua B&B và GA (chậm)
--skip bb ga

# Chỉ chạy vài solver
--only dp greedy gnn

# Đánh giá GNN với Conflict Graph
--gnn_graph_type conflict_static

# Giới hạn số instance (chạy thử nhanh)
--n 20

# B&B timeout per instance
--bb_timeout 30
```

---

## 10. Vẽ biểu đồ

```bash
python src/scripts/plot_results.py --results_dir results/compare --out_dir plots/
```

---

## 11. Thí nghiệm nâng cao

### Cross-scale generalization

Đánh giá khả năng tổng quát hoá khi test trên kích thước instance khác nhau:

```bash
python src/scripts/cross_scale_eval.py \
    --model_type gnn \
    --model_path results/GNN/gnn_best.pt \
    --test_sets data/knapsack_ilp/test data/pisinger/type_03/test \
    --labels "Uncorrelated" "Strongly Corr" \
    --include_baselines \
    --skip_bb \
    --out_dir results/cross_scale
```

### Benchmark Pisinger hard sets

```bash
python src/scripts/Benchmark_Hard.py --types 3 5 6 --n_items 100
```

---

## 12. Streamlit Dashboard

Giao diện trực quan để so sánh kết quả, xem biểu đồ, và phân tích từng instance:

```bash
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`.

---

## 13. Google Colab

Để train trên GPU miễn phí (T4), upload toàn bộ project lên Google Drive rồi mở file `GNNForKnapSack_Colab.ipynb` trong Colab.

**Thời gian ước tính trên T4 GPU:**

| Model | Ước tính |
|---|---|
| GNN supervised (150 epochs) | 20–40 phút |
| S2V-DQN (50 000 steps) | 30–60 phút |
| DQN (50 000 steps) | 20–40 phút |
| REINFORCE (50 epochs) | 20–40 phút |
| **Tổng** | **~2–4 giờ** |

---

## 14. Troubleshooting

| Lỗi | Nguyên nhân | Cách khắc phục |
|---|---|---|
| `FileNotFoundError: No instance_*.npz` | Chưa sinh dữ liệu | Chạy `Generate_Data.py` |
| `EOFError` khi load checkpoint | File `.pt` rỗng hoặc bị corrupt | Train lại hoặc dùng file `_best.pt` |
| Feature dim mismatch (6 vs 7) | Cache dataset cũ | Xóa `processed_dataset.pt` |
| BB chạy rất lâu | B&B bùng nổ tổ hợp | Thêm `--bb_timeout 30` hoặc `--skip bb` |
| `AssertionError: Torch not compiled with CUDA` | Máy không có GPU NVIDIA | Dùng `--device cpu` hoặc train trên Colab |
| `ModuleNotFoundError: GNNForKnapSack` | PYTHONPATH chưa đúng | Chạy lệnh từ thư mục gốc project |

---

## 15. Ghi chú kỹ thuật

### 15.1 Kiến trúc GNN

- **Node features**: 7 chiều (`w_norm`, `v_norm`, `ratio_norm`, `cap_ratio`, `cap_util`, `item_frac`, `w_vs_mean`)
- **Graph builder**: hỗ trợ KNN, Conflict Static, Random, Full
- **Convolution**: GINConv (mặc định) hoặc SAGEConv (`conv_type: sage`)
- **Normalization**: LayerNorm (ổn định với batch đa kích thước)
- **Global context**: `GlobalContextInjection` inject thông tin toàn cục vào mỗi node

### 15.2 Decode logic

- `greedy_feasible`: sắp xếp theo probability, lấy đến khi đầy túi — đảm bảo feasible
- `dp_subset_decode`: lấy top-m item theo prob, chạy DP trên tập con — tối ưu hơn
- `beam_search`: beam search trên không gian lời giải

### 15.3 Device flexibility

Tất cả script training và evaluation hỗ trợ `--device auto|cpu|cuda`. Config YAML cũng có trường `device: "auto"`. Thứ tự ưu tiên: CLI `--device` > config YAML > mặc định `"auto"`.

### 15.4 Capacity generation

Capacity của instance thứ $i$ trong $N$ instance:

$$C_i = \frac{i+1}{N+1} \cdot \sum_j w_j$$

---

## Liên hệ và đóng góp

Chúc may mắn với nghiên cứu!
