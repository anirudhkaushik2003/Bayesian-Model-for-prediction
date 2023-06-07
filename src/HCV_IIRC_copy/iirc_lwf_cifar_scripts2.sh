### LwF + SPL
python ./prior_main_ptm_training_icarl.py \
    --method learning_without_forgetting_ptm \
    --n_memories_per_class -1 \
    --group ISIC_EXPL_TEST \
    --save_each_task_model \
    --epochs_per_task 14 \
    --total_n_memories -1 \
    --use_best_model \
    --checkpoint_interval 5 \
    --wandb_project ISIC_BENCHMARK_EXPL_4 \
    --tasks_configuration_id 2 \
    --batch_size 64 \
    --lr 0.001 \
    --threshold 0.58

### LwF
python ./prior_main.py \
    --method learning_without_forgetting \
    --n_memories_per_class -1 \
    --group ISIC_EXPL_TEST \
    --save_each_task_model \
    --epochs_per_task 14 \
    --total_n_memories -1 \
    --use_best_model \
    --checkpoint_interval 5 \
    --wandb_project ISIC_BENCHMARK_EXPL_4 \
    --tasks_configuration_id 2 \
    --batch_size 64 \
    --lr 0.001
    

### Joint training
python ./prior_main.py \
    --method finetune \
    --n_memories_per_class -1 \
    --group ISIC_EXPL_TEST \
    --reduce_lr_on_plateau \
    --save_each_task_model \
    --complete_info \
    --use_best_model \
    --incremental_joint \
    --total_n_memories -1 \
    --n_memories_per_class 20 \
    --reduce_lr_on_plateau \
    --group ISIC_EXPL_TEST \
    --save_each_task_model \
    --epochs_per_task 14 \
    --total_n_memories -1 \
    --use_best_model \
    --wandb_project ISIC_BENCHMARK_EXPL_4 \
    --tasks_configuration_id 2 \
    --batch_size 64 \
    --lr 0.001
    

### FT
python ./prior_main.py \
    --method finetune \
    --n_memories_per_class -1 \
    --group ISIC_EXPL_TEST \
    --save_each_task_model \
    --epochs_per_task 14 \
    --total_n_memories -1 \
    --use_best_model \
    --checkpoint_interval 5 \
    --wandb_project ISIC_BENCHMARK_EXPL_4 \
    --tasks_configuration_id 2 \
    --batch_size 64 \
    --lr 0.001
    