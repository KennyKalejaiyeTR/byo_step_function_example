from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score, recall_at_k
import numpy as np
import scipy
import joblib
from datetime import datetime
import argparse

from smexperiments.trial import Trial
from smexperiments.tracker import Tracker


def evaluate_model(model, test, train, k=10):
    
    train_precision = precision_at_k(model, train, k=k).mean()
    test_precision = precision_at_k(model, test, k=k, train_interactions=train).mean()
    
    train_recall = recall_at_k(model, train, k=k).mean()
    test_recall = recall_at_k(model, test, k=k, train_interactions=train).mean()

    train_auc = auc_score(model, train).mean()
    test_auc = auc_score(model, test, train_interactions=train).mean()
    
    return dict(train_precision = float(train_precision),
               test_precision = float(test_precision),
               train_recall = float(train_recall),
               test_recall = float(test_recall),
               train_auc = float(train_auc),
               test_auc = float(test_auc)
               )


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/opt/ml/processing/input/")
    parser.add_argument("--output_path", type=str, default="/opt/ml/processing/output/")
    parser.add_argument("--model_path", type=str, default="/opt/ml/processing/model/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--loss_function", type=str, default="bpr")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--trial_name", type=str, default="testing")
    args, _ = parser.parse_known_args()

    # load training data
    train = scipy.sparse.load_npz(f"{args.input_path}/train.npz")

    # load testing data
    test = scipy.sparse.load_npz(f"{args.input_path}/test.npz")

    # load model
    model = joblib.load(f"{args.model_path}/model.gz")

    # evaluate model
    evaluation = evaluate_model(model, test, train, k=args.k)

    # log results
    trial = Trial.load(args.trial_name)
    tracker = Tracker.create("evaluation")
    trial.add_trial_component(tracker)
    evaluation.update({"lr": args.lr, 
                        "epochs": args.epochs, 
                        "loss_function": args.loss_function,
                        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
    tracker.log_parameters(evaluation)
    tracker.close()
