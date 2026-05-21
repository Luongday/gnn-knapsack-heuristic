import pandas as pd
import glob


def compute_stats(file_path):
    """Đọc file CSV và tính các chỉ số thống kê."""
    try:
        df = pd.read_csv(file_path)
        n_samples = len(df)

        # Mean ratio (trung bình cộng của các ratio)
        mean_ratio = df['ratio'].mean() if 'ratio' in df.columns else None

        # Overall ratio (tổng giá trị / tổng tối ưu)
        if 'total_value' in df.columns and 'dp_value' in df.columns:
            overall_ratio = df['total_value'].sum() / df['dp_value'].sum()
        else:
            overall_ratio = None

        # Mean inference time (ms)
        mean_inference_time = df['inference_time_ms'].mean() if 'inference_time_ms' in df.columns else None

        # Kiểm tra feasibility (nếu cột feasible tồn tại)
        all_feasible = df['feasible'].all() if 'feasible' in df.columns else None

        # In kết quả
        print(f"📄 File: {file_path}")
        print(f"   Số mẫu: {n_samples}")
        if mean_ratio is not None:
            print(f"   Mean Ratio (trung bình các ratio): {mean_ratio:.6f}")
        if overall_ratio is not None:
            print(f"   Overall Ratio (∑value/∑dp): {overall_ratio:.6f}")
        if mean_inference_time is not None:
            print(f"   Mean Inference Time: {mean_inference_time:.2f} ms")
        if all_feasible is not None:
            print(f"   Tất cả feasible: {all_feasible}")
        print()
        return mean_ratio, mean_inference_time, overall_ratio
    except Exception as e:
        print(f"❌ Lỗi khi đọc {file_path}: {e}")
        return None, None, None


if __name__ == "__main__":
    # Liệt kê các file bạn muốn tính toán (có thể dùng glob để tự động tìm)
    file_list = [
        "results/cross_scale/s2v_conflict_on_small.csv",
        "results/cross_scale/s2v_conflict_on_n100.csv",
        "results/cross_scale/s2v_conflict_on_n200.csv",

        "results/cross_scale/s2v_knn_on_small.csv",
        "results/cross_scale/s2v_knn_on_n100.csv",
        "results/cross_scale/s2v_knn_on_n200.csv",

        "results/cross_scale/s2v_full_on_small.csv",
        "results/cross_scale/s2v_full_on_n100.csv",
        "results/cross_scale/s2v_full_on_n200.csv",

        "results/cross_scale/s2v_random_on_small.csv",
        "results/cross_scale/s2v_random_on_n100.csv",
        "results/cross_scale/s2v_random_on_n200.csv",

        "results/cross_scale/dqn_on_small.csv",
        "results/cross_scale/dqn_on_n100.csv",
        "results/cross_scale/dqn_on_n200.csv",

        "results/cross_scale/reinforce_hard_on_small.csv",
        "results/cross_scale/reinforce_hard_on_n100.csv",
        "results/cross_scale/reinforce_hard_on_n200.csv",

        "results/cross_scale/reinforce_none_on_small.csv",
        "results/cross_scale/reinforce_none_on_n100.csv",
        "results/cross_scale/reinforce_none_on_n200.csv",

        "results/cross_scale/reinforce_polyak_on_small.csv",
        "results/cross_scale/reinforce_polyak_on_n100.csv",
        "results/cross_scale/reinforce_polyak_on_n200.csv",
    ]

    # Hoặc tự động lấy tất cả file CSV trong thư mục hiện tại:
    # file_list = glob.glob("*.csv")

    for file_path in file_list:
        compute_stats(file_path)