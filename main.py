from train_and_predict import model_train, model_predict
from model_parameters import settings


def main():
    if settings["mode"] == "train":
        model_train()
    else:
        model_predict(settings["file_name"])


if __name__ == "__main__":
    main()
