from nn_model.train_and_predict import model_train, predict_image
from settings import settings


def run():
    if settings["mode"] == "train":
        model_train()
    else:
        labels = predict_image(settings["data_path"], settings["file_name"])


if __name__ == "__main__":
    run()
