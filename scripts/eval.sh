source .venv/bin/activate

#========== PPO: ==========
# path="/home/jovyan/avarlamov/skoltech/super-mario-project/checkpoints/mario_lr1e-5/update_8750_x3161.0.pt"

path="/home/jovyan/avarlamov/skoltech/super-mario-project/checkpoints/a3c_mario_lr1e-5/update_4000_x559.8.pt"

# Extract the second-to-last directory in the path (pred-last part before the file)
pred_last_part=$(basename "$(dirname "$path")")
log_dir_base="results/${pred_last_part}_$(basename "$path" .pt)"

# Stochastic
log_dir_stochastic="${log_dir_base}_stochastic"
python test.py $path --episodes 20 --gif-path eval_result_stochastic.gif --log-dir $log_dir_stochastic --stochastic

# Deterministic
log_dir_deterministic="${log_dir_base}_deterministic"
python test.py $path --episodes 20 --gif-path eval_result_deterministic.gif --log-dir $log_dir_deterministic


# =========== A2C: ==========

path=

# Extract the second-to-last directory in the path (pred-last part before the file)
pred_last_part=$(basename "$(dirname "$path")")
log_dir_base="results/${pred_last_part}_$(basename "$path" .pt)"

# Stochastic
log_dir_stochastic="${log_dir_base}_stochastic"
python test.py $path --episodes 20 --gif-path eval_result_stochastic.gif --log-dir $log_dir_stochastic --stochastic

# Deterministic
log_dir_deterministic="${log_dir_base}_deterministic"
python test.py $path --episodes 20 --gif-path eval_result_deterministic.gif --log-dir $log_dir_deterministic

# =========== REINFORCE: ==========

path=

# Extract the second-to-last directory in the path (pred-last part before the file)
pred_last_part=$(basename "$(dirname "$path")")
log_dir_base="results/${pred_last_part}_$(basename "$path" .pt)"

# Stochastic
log_dir_stochastic="${log_dir_base}_stochastic"
python test.py $path --episodes 20 --gif-path eval_result_stochastic.gif --log-dir $log_dir_stochastic --stochastic

# Deterministic
log_dir_deterministic="${log_dir_base}_deterministic"
python test.py $path --episodes 20 --gif-path eval_result_deterministic.gif --log-dir $log_dir_deterministic