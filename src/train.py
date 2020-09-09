from lightfm import LightFM
import numpy as np
import scipy
import joblib
import argparse


def train_model(data, epochs, lr, loss):
    
    model = LightFM(learning_rate=lr, loss=loss)
    model.fit_partial(data, epochs=epochs)
    
    return model


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/opt/ml/processing/input/")
    parser.add_argument("--output_path", type=str, default="/opt/ml/processing/output/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--loss_function", type=str, default="bpr")
    args, _ = parser.parse_known_args()

    # load training data
    train = scipy.sparse.load_npz(f"{args.input_path}/train.npz")

    # train model
    model = train_model(train, args.epochs, args.lr, args.loss_function)

    # save model
    joblib.dump(model, f"{args.output_path}/model.gz")
