# Self-BackTracking

## training
python train.py --config configs/sft.conf --resume --wandb

## Inference
python eval_search.py --ckpt model_ckpt_path --data train_backtrack.json --decoder self_backtrack --k 16 --backtrack_times 1

## Self-improvement
python train_self_improvement.py 