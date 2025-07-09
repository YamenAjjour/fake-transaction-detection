from datasets import Dataset, DatasetDict
import datasets
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW

from torch import nn
def load_dataset() -> DatasetDict:
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    train_dataset = Dataset.from_pandas(df_train)
    test_dataset = Dataset.from_pandas(df_test)
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
    return dataset

def format_product_description(data_point):
    data_point["input"] = data_point["product_description"] + "[cls]"
    return  data_point

def get_tokenizer(max_length):
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    def tokenize_project_description(data_point):
        data_point = tokenizer(data_point["input"], truncation=True, padding=True, max_length=max_length)
        return data_point
    return tokenize_project_description

def train(learning_rate, batch, epochs) -> AutoModel:
    dataset = load_dataset()
    tokenizer = get_tokenizer(100)
    dataset["train"].map(lambda x: print(x))
    dataset["train"].map(lambda x: tokenizer)
    train_data_loader = DataLoader(dataset["train"], batch_size=batch)
    test_data_loader = DataLoader(dataset["test"], batch_size=batch)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=2)
    optimizer = AdamW(model.parameters(),learning_rate=learning_rate)
    criterion = nn.CrossEntropyLoss().cuda()
    train_loss = 0
    for epoch in range(epochs):
        for _, batch in enumerate(train_data_loader):
            mask = batch["attention_mask"].to("cuda")
            input_ids = batch["input_ids"].to("cuda")
            output = model(input_ids, mask)

            batch_loss = criterion(output[0], batch["Fake"].cuda())
            train_loss += batch_loss
            batch_loss.backword()
            optimizer.step()
            optimizer.zero_grad()


    return model
