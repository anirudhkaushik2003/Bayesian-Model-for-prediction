# ### LwF + SPL + RRR
python ./prior_main_ptm_training_RRR.py \
    --method learning_without_forgetting_ptm_RRR \
    --n_memories_per_class -1 \
    --saliency_n_memories_per_class 20 \
    --group ISIC_EXPL_RRR \
    --save_each_task_model \
    --epochs_per_task 14 \
    --total_n_memories -1 \
    --saliency_total_n_memories -1 \
    --use_best_model \
    --checkpoint_interval 5 \
    --wandb_project ISIC_BENCHMARK_EXPL_5 \
    --tasks_configuration_id 3 \
    --batch_size 64 \
    --lr 0.001 \
    --threshold 0.58


    