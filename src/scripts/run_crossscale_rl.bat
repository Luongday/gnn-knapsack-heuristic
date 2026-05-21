@echo off
setlocal

REM S2V-DQN: 4 graph types x 3 test sets = 12 runs
REM REINFORCE: knn dai dien x 3 test sets = 3 runs

REM ========== REINFORCE: knn (dai dien) ==========
echo [REINFORCE-knn] small...
python src\rl\REINFORCE\Evaluate_reinforce.py ^
    --dataset_dir data\cross_scale\test_small ^
    --model_path results\GNN_REINFORCE\gnn_reinforce_knn_none\gnn_reinforce_knn_none.pt ^
    --graph_type knn ^
    --out_csv results\cross_scale\reinforce_none_on_small.csv

echo [REINFORCE-knn] n100...
python src\rl\REINFORCE\Evaluate_reinforce.py ^
    --dataset_dir data\cross_scale\test_n100 ^
    --model_path results\GNN_REINFORCE\gnn_reinforce_knn_none\gnn_reinforce_knn_none.pt ^
    --graph_type knn ^
    --out_csv results\cross_scale\reinforce_none_on_n100.csv

echo [REINFORCE-knn] n200...
python src\rl\REINFORCE\Evaluate_reinforce.py ^
    --dataset_dir data\cross_scale\test_n200 ^
    --model_path results\GNN_REINFORCE\gnn_reinforce_knn_none\gnn_reinforce_knn_none.pt ^
    --graph_type knn ^
    --out_csv results\cross_scale\reinforce_none_on_n200.csv

echo [REINFORCE-knn] small...
python src\rl\REINFORCE\Evaluate_reinforce.py ^
    --dataset_dir data\cross_scale\test_small ^
    --model_path results\GNN_REINFORCE\gnn_reinforce_knn_hard\gnn_reinforce_knn_hard.pt ^
    --graph_type knn ^
    --out_csv results\cross_scale\reinforce_hard_on_small.csv

echo [REINFORCE-knn] n100...
python src\rl\REINFORCE\Evaluate_reinforce.py ^
    --dataset_dir data\cross_scale\test_n100 ^
    --model_path results\GNN_REINFORCE\gnn_reinforce_knn_hard\gnn_reinforce_knn_hard.pt ^
    --graph_type knn ^
    --out_csv results\cross_scale\reinforce_hard_on_n100.csv

echo [REINFORCE-knn] n200...
python src\rl\REINFORCE\Evaluate_reinforce.py ^
    --dataset_dir data\cross_scale\test_n200 ^
    --model_path results\GNN_REINFORCE\gnn_reinforce_knn_hard\gnn_reinforce_knn_hard.pt ^
    --graph_type knn ^
    --out_csv results\cross_scale\reinforce_hard_on_n200.csv

echo [REINFORCE-knn] small...
python src\rl\REINFORCE\Evaluate_reinforce.py ^
    --dataset_dir data\cross_scale\test_small ^
    --model_path results\GNN_REINFORCE\gnn_reinforce_knn_polyak\gnn_reinforce_knn_polyak.pt ^
    --graph_type knn ^
    --out_csv results\cross_scale\reinforce_polyak_on_small.csv

echo [REINFORCE-knn] n100...
python src\rl\REINFORCE\Evaluate_reinforce.py ^
    --dataset_dir data\cross_scale\test_n100 ^
    --model_path results\GNN_REINFORCE\gnn_reinforce_knn_polyak\gnn_reinforce_knn_polyak.pt ^
    --graph_type knn ^
    --out_csv results\cross_scale\reinforce_polyak_on_n100.csv

echo [REINFORCE-knn] n200...
python src\rl\REINFORCE\Evaluate_reinforce.py ^
    --dataset_dir data\cross_scale\test_n200 ^
    --model_path results\GNN_REINFORCE\gnn_reinforce_knn_polyak\gnn_reinforce_knn_polyak.pt ^
    --graph_type knn ^
    --out_csv results\cross_scale\reinforce_polyak_on_n200.csv

echo.
echo Xong. 15 runs hoan thanh.
pause
