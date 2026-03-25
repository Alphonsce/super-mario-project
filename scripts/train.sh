source .venv/bin/activate

# ===== PPO: ========

# CUDA_VISIBLE_DEVICES=0 python train.py --world 1 --stage 1 --lr 1e-4 --experiment mario_lr1e-4
# CUDA_VISIBLE_DEVICES=1 python train.py --world 1 --stage 1 --lr 3e-4 --experiment mario_lr3e-4
# CUDA_VISIBLE_DEVICES=2 python train.py --world 1 --stage 1 --lr 1e-3 --experiment mario_lr1e-3
# CUDA_VISIBLE_DEVICES=3 python train.py --world 1 --stage 1 --lr 3e-3 --experiment mario_lr3e-3

# CUDA_VISIBLE_DEVICES=3 python ppo_train.py --world 1 --stage 1 --lr 5e-5 --experiment mario_lr5e-5

CUDA_VISIBLE_DEVICES=0 python ppo_train.py --world 1 --stage 1 --lr 1e-5 --experiment ppo_mario_lr1e-5_complex_action --action_type complex

# ==== REINFORCE: ====

# CUDA_VISIBLE_DEVICES=0 python reinforce_train.py --world 1 --stage 1 --lr 1e-4 --experiment reinforce_mario_lr1e-4
# CUDA_VISIBLE_DEVICES=1 python reinforce_train.py --world 1 --stage 1 --lr 3e-4 --experiment reinforce_mario_lr3e-4
# CUDA_VISIBLE_DEVICES=2 python reinforce_train.py --world 1 --stage 1 --lr 1e-5 --experiment reinforce_mario_lr1e-5
# CUDA_VISIBLE_DEVICES=3 python reinforce_train.py --world 1 --stage 1 --lr 5e-5 --experiment reinforce_mario_lr5e-5

# ===== A3C: ======

# CUDA_VISIBLE_DEVICES=4 python a3c_train.py --world 1 --stage 1 --lr 1e-4 --experiment a3c_mario_lr1e-4
# CUDA_VISIBLE_DEVICES=5 python a3c_train.py --world 1 --stage 1 --lr 3e-4 --experiment a3c_mario_lr3e-4
# CUDA_VISIBLE_DEVICES=6 python a3c_train.py --world 1 --stage 1 --lr 1e-5 --experiment a3c_mario_lr1e-5
# CUDA_VISIBLE_DEVICES=7 python a3c_train.py --world 1 --stage 1 --lr 5e-5 --experiment a3c_mario_lr5e-5


# ======== A2C: ========

# CUDA_VISIBLE_DEVICES=0 python a2c_train.py --world 1 --stage 1 --lr 1e-4 --experiment a2c_mario_lr1e-4
# CUDA_VISIBLE_DEVICES=1 python a2c_train.py --world 1 --stage 1 --lr 3e-4 --experiment a2c_mario_lr3e-4
# CUDA_VISIBLE_DEVICES=2 python a2c_train.py --world 1 --stage 1 --lr 1e-5 --experiment a2c_mario_lr1e-5
# CUDA_VISIBLE_DEVICES=3 python a2c_train.py --world 1 --stage 1 --lr 5e-5 --experiment a2c_mario_lr5e-5

# CUDA_VISIBLE_DEVICES=4 python a2c_train.py --world 1 --stage 1 --lr 1e-3 --experiment a2c_mario_lr1e-3