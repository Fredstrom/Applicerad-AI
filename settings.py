settings = {
    # This setting decides if we run model_train() or model_predict() -> train/test
    "mode": "train",

    # File Paths:
    "api_folder": "application/static/api_folder/",
    "data_path": "data_jpg",
    "test_path": "nn_model\\data_jpg\\test\\",
    "model_weights": "nn_model\\Model_results\\model_91.pth",

    # For Training:
    "learning_rate": 0.00027,
    "weight_decay": 0.1,
    "batch_size": 20,
    "num_epochs": 25,
    "image_size": (600, 600),
}
