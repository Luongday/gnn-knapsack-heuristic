@echo off
setlocal

set TOPM_SMALL=30
set TOPM_N100=80
set TOPM_N200=150
set BEAM_WIDTH=5

if not exist "results\cross_scale" mkdir "results\cross_scale"
REM Loop 4 models × 3 test sets × 3 decode methods = 36 runs
for %%M in (knn conflict_static random full) do (
    echo ==================================================
    echo Model: %%M
    echo ==================================================

    :: REM === Greedy decode ===
    python src\gnn\Evaluate_GNN.py ^
        --dataset_dir data\cross_scale\test_small ^
        --model_path results\ablation\gnn_%%M\gnn_best.pt ^
        --graph_type %%M ^
        --out_csv results\cross_eval\%%M_on_small_greedy.csv

    python src\gnn\Evaluate_GNN.py ^
        --dataset_dir data\cross_scale\test_n100 ^
        --model_path results\ablation\gnn_%%M\gnn_best.pt ^
        --graph_type %%M ^
        --out_csv results\cross_eval\%%M_on_n100_greedy.csv

    python src\gnn\Evaluate_GNN.py ^
        --dataset_dir data\cross_scale\test_n200 ^
        --model_path results\ablation\gnn_%%M\gnn_best.pt ^
        --graph_type %%M ^
        --out_csv results\cross_eval\%%M_on_n200_greedy.csv

    :: REM === DP subset decode ===
    python src\gnn\Evaluate_GNN_DP.py ^
        --dataset_dir data\cross_scale\test_small ^
        --model_path results\ablation\gnn_%%M\gnn_best.pt ^
        --graph_type %%M ^
        --top_m %TOPM_SMALL% ^
        --out_csv results\cross_eval\%%M_on_small_dp.csv

    python src\gnn\Evaluate_GNN_DP.py ^
        --dataset_dir data\cross_scale\test_n100 ^
        --model_path results\ablation\gnn_%%M\gnn_best.pt ^
        --graph_type %%M ^
        --top_m %TOPM_N100% ^
        --out_csv results\cross_eval\%%M_on_n100_dp.csv

    python src\gnn\Evaluate_GNN_DP.py ^
        --dataset_dir data\cross_scale\test_n200 ^
        --model_path results\ablation\gnn_%%M\gnn_best.pt ^
        --graph_type %%M ^
        --top_m %TOPM_N200% ^
        --out_csv results\cross_eval\%%M_on_n200_dp.csv

    :: REM === Beam search decode ===
    python src\gnn\Evaluate_GNN.py ^
        --dataset_dir data\cross_scale\test_small ^
        --model_path results\ablation\gnn_%%M\gnn_best.pt ^
        --graph_type %%M ^
        --decode_strategy beam_search ^
        --beam_width %BEAM_WIDTH% ^
        --out_csv results\cross_eval\%%M_on_small_beam.csv

    python src\gnn\Evaluate_GNN.py ^
        --dataset_dir data\cross_scale\test_n100 ^
        --model_path results\ablation\gnn_%%M\gnn_best.pt ^
        --graph_type %%M ^
        --decode_strategy beam_search ^
        --beam_width %BEAM_WIDTH% ^
        --out_csv results\cross_eval\%%M_on_n100_beam.csv

    python src\gnn\Evaluate_GNN.py ^
        --dataset_dir data\cross_scale\test_n200 ^
        --model_path results\ablation\gnn_%%M\gnn_best.pt ^
        --graph_type %%M ^
        --decode_strategy beam_search ^
        --beam_width %BEAM_WIDTH% ^
        --out_csv results\cross_eval\%%M_on_n200_beam.csv
)

echo.
echo All 36 evaluations complete.
pause