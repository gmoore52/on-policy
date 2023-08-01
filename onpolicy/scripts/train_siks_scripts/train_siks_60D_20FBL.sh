#!/bin/sh
env="SIkS"
algo="rmappo"
exp="coverage"
map="60D_20FBL_5S"
seed_max=3

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_siks.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} \
    --filename ${map}.json --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 1 --episode_length 400 \
    --num_env_steps 60000 --ppo_epoch 15 --use_value_active_masks --use_eval --eval_episodes 32 --n_agents 5 --use_feature_normalization true \
    # --randomly_deploy
done
