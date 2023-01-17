from nn_model.train_and_predict import model_train, predict_image, predict_folder
from settings import settings


def run():
    if settings["mode"] == "train":
        model_train()
    else:
        settings["model_weights"] = f"../{settings['model_weights']}"
        predict_folder(settings["test_path"])


if __name__ == "__main__":
    run()
