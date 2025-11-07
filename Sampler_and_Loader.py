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

# The sampler needs a dataset with a "get_labels" method. Check the code if you have any doubt!
test_dataset.get_labels = lambda:[instance[1] for instance in test_dataset]  # NO FLAT CHARACTER IMAGES
test_sampler = TaskSampler(test_dataset, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS)

test_loader = DataLoader(
    test_dataset,
    batch_sampler=test_sampler,
    num_workers=8,  # from 12.
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,
)

(
    example_support_images,
    example_support_labels,
    example_query_images,
    example_query_labels,
    example_class_ids,
) = next(iter(test_loader))

plot_images(example_support_images, "support images", images_per_row=N_SHOT)
plot_images(example_query_images, "query images", images_per_row=N_QUERY)
