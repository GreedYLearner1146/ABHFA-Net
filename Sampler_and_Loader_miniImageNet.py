train_dataset.get_labels = lambda: [instance[1] for instance in train_dataset]  # AIDER/CDD would work since they have the same train_dataset naming 
                                                                                # as in AIDER_dataloader.py and CDD_dataloader.py

train_sampler = TaskSampler(
    train_dataset, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_TRAINING_EPISODES
)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    num_workers=8,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,
)
