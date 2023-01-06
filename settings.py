settings = {
    # This setting decides if we run model_train() or model_predict() -> train/test
    "mode": "test",

    # File Paths:
    "data_path": "nn_model\\data_jpg",
    "test_path": "nn_model\\data_jpg\\test\\",
    "model_weights": "nn_model\\Model_results\\model_95.pth",

    # For Training:
    "learning_rate": 0.00257,
    "weight_decay": 0.1,
    "batch_size": 20,
    "num_epochs": 25,
    "image_size": (400, 400),

    # For testing:
    "file_name": "*",  # * = ALL images in test_folder, if you want a specific image, enter filename here.
}
