from datasets import Dataset, DatasetDict
import datasets
import pandas as pd
import argparse
import torch
import optuna
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from config import *

def load_dataset(debug) -> DatasetDict:
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    df_val = df_train.sample(len(df_test))
    df_train = df_train[~df_train.index.isin(df_val.index)]
    if debug:
        df_train = df_train.sample(50)
        df_test = df_test.sample(50)
    size_non_fake = len(df_train[df_train["Fake"]==0]) - len(df_train[df_train["Fake"]==1])
    df_to_add = df_test[df_test["Fake"]==1].sample(n=size_non_fake, replace=True)
    df_train = pd.concat([df_to_add, df_train])
    train_dataset = Dataset.from_pandas(df_train[["Fake", "product_description"]].rename(columns={"Fake":"output"})).to_iterable_dataset()
    val_dataset = Dataset.from_pandas(df_val[["Fake", "product_description"]].rename(columns={"Fake":"output"})).to_iterable_dataset()
    test_dataset = Dataset.from_pandas(df_test[["Fake", "product_description"]].rename(columns={"Fake":"output"})).to_iterable_dataset()
    dataset = DatasetDict({"train": train_dataset, "val" : val_dataset, "test": test_dataset})
    return dataset

def format_product_description(data_point):
    data_point["input"] = "[cls]" + data_point["product_description"]
    return  data_point

def get_tokenizer(max_length):
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased", truncation=True, max_length=max_length, padding=True)
    def tokenize_project_description(data_point):
        data_point = tokenizer(data_point["input"], truncation=True, max_length=max_length, return_tensors="pt", padding="max_length")

        return data_point
    return tokenize_project_description

dataset = load_dataset(True)

def train(trial =None, learning_rate : float =None , batch : int = None, epochs : int = None, debug=False) -> AutoModel:
    if trial:
        learning_rate = trial.suggest_float("x", 1e-5, 5e-5)
        batch = trial.suggest_int("batch-size", 4, 16)
        epochs = trial.suggest_int("epochs", 1 , 10)
        test_split="val"
    else:
        test_split = "test"
    print(f"testing {learning_rate}, batch {batch}, epochs {epochs}")

    tokenizer = get_tokenizer(100)

    dataset["train"]=dataset["train"].map(format_product_description).map(tokenizer)
    dataset[test_split]=dataset[test_split].map(format_product_description).map(tokenizer)
    train_data_loader = DataLoader(dataset["train"], batch_size=batch)
    val_data_loader = DataLoader(dataset[test_split], batch_size=batch)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=2).to("cuda")
    optimizer = AdamW(model.parameters(),lr=learning_rate)
    criterion = nn.CrossEntropyLoss().cuda()
    config = get_config()
    metrics = {}
    for epoch in range(epochs):
        train_loss = 0
        for step, batch in enumerate(train_data_loader):
            mask = batch["attention_mask"].to("cuda").squeeze(1)
            input_ids = batch["input_ids"].to("cuda").squeeze(1)
            output = model(input_ids, mask)

            batch_loss = criterion(output[0], batch["output"].cuda())
            train_loss += batch_loss
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metrics["train/loss"] = train_loss / (step +1)
        val_loss = 0
        print(f"train_loss={metrics["train/loss"]}")
        labels = []
        predictions = []
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_data_loader):
                mask = batch["attention_mask"].to("cuda").squeeze(1)
                input_ids = batch["input_ids"].to("cuda").squeeze(1)
                output = model(input_ids, mask)

                batch_loss = criterion(output[0], batch["output"].cuda())
                val_loss += batch_loss
                metrics[f"{test_split}/loss"] = val_loss / (step+1)
                labels.extend(batch["output"].cpu().tolist())
                predictions.extend(torch.argmax(output[0],dim=1).cpu().tolist())
        metrics["f1-score fake"] = f1_score(labels, predictions, labels=[1])
        metrics["accuracy"] = accuracy_score(labels, predictions)

        print(f"f1={metrics["f1-score fake"]}")
        print(f"accuracy={metrics["accuracy"]}")

        print(f"{test_split}_loss={metrics[f"{test_split}/loss"]}")
    if not trial:
        model.save_pretrained(config["model_path"])

    return metrics["f1-score fake"]

def load_modal_and_tokenizer():
    config = get_config()
    model_path = config["model_path"]
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased", truncation=True, max_length=100, padding=True)
    return model, tokenizer

def predict(transaction, model, tokenizer):
    instance = tokenizer(transaction, padding="max_length", truncation=True, return_tensors="pt")
    output = model(instance["input_ids"].to("cuda"), instance["attention_mask"].to("cuda"))
    i = torch.argmax(output[0]).item()
    map = {0: "not fake", 1: "fake"}
    return map[i]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--optimize", action="store_true")

    args = parser.parse_args()
    if args.optimize:
        study = optuna.create_study()
        study.optimize(train, n_trials=10, direction=["maximize"])
        print("best params are")
        print(study.best_params)
        print("best performance is")
        print(study.best_value)
    train(learning_rate=args.learning_rate, batch=args.batch_size,epochs= args.epochs,debug= args.debug)
