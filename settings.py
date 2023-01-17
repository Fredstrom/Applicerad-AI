settings = {
    # This setting decides if we run model_train() or model_predict() -> train/test
    "mode": "train",

    # File Paths:
    "api_folder": "application/static/api_folder/",
    "data_path": "data_jpg",
    "test_path": "nn_model\\data_jpg\\test\\",
    "model_weights": "nn_model\\Model_results\\model_91.pth",
    "log_path": "nn_model/logs/",
    # For Training:
    "learning_rate": 0.00020,
    "weight_decay": 0.15,
    "batch_size": 24,
    "num_epochs": 35,
    "image_size": (600, 600),
}
