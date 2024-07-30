import sys
import os
from time import sleep
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import cohen_kappa_score
import config
import train
from utils import (
    save_checkpoint,
    load_checkpoint,
    check_accuracy,
    Slowprint,
    ScoreLogFile,
    GetPreviousScore,
)


def make_prediction(model, loader, file):
    preds = []
    filenames = []
    model.eval()

    loop = tqdm(loader, desc="Making Predictions: ", leave=False, ascii=config.ASCII,
                delay=config.DELAYTIME, colour=config.COLOR, bar_format=config.BARFORMAT)
    count = 0
    for x, y, files in loop:
        count += 1
        x = x.to(config.DEVICE)
        with torch.no_grad():
            predictions = model(x)
            # Convert MSE floats to integer predictions
            predictions[predictions < 0.5] = 0
            predictions[(predictions >= 0.5) & (predictions < 1.5)] = 1
            predictions[(predictions >= 1.5) & (predictions < 2.5)] = 2
            predictions[(predictions >= 2.5) & (predictions < 3.5)] = 3
            predictions[(predictions >= 3.5) & (
                predictions < 1000000000000)] = 4
            predictions = predictions.long().view(-1)
            y = y.view(-1)

            preds.append(predictions.cpu().numpy())
            filenames += map(list, zip(files[0], files[1]))
        if int(loop.total) == count:
            loop.colour = config.COLOR_COMPLETE
            loop.update()

    filenames = [item for sublist in filenames for item in sublist]
    df = pd.DataFrame(
        {"image": filenames, "level": np.concatenate(preds, axis=0)})
    Slowprint("- Writing data to csv file...")
    df.to_csv(file, index=False)
    model.train()
    sys.stdout.write('\x1b[1A')
    sys.stdout.write('\x1b[2K')
    Slowprint("- Done with predictions.")


class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.csv = pd.read_csv(csv_file)

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        example = self.csv.iloc[index, :]
        features = example.iloc[: example.shape[0] -
                                4].to_numpy().astype(np.float32)
        labels = example.iloc[-4:-2].to_numpy().astype(np.int64)
        filenames = example.iloc[-2:].values.tolist()
        return features, labels, filenames


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d((1536 + 1) * 2),
            nn.Linear((1536+1) * 2, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 2),
        )

    def forward(self, x):
        return self.model(x)


def Train_blend():
    Slowprint("Setting up model and dataset...")
    model = MyModel().to(config.DEVICE)
    ds = MyDataset(csv_file=config.WORKINGDIR+"/train_set/train_blend.csv")
    loader = DataLoader(
        ds, batch_size=config.BATCH_SIZE_CHECKACC, num_workers=3, pin_memory=True, shuffle=True
    )
    ds_test_public = MyDataset(
        csv_file=config.WORKINGDIR+"/test_set/test_blend(public).csv")
    loader_test_public = DataLoader(
        ds_test_public, batch_size=config.BATCH_SIZE_CHECKACC, num_workers=2, pin_memory=True, shuffle=False
    )
    ds_test_private = MyDataset(
        csv_file=config.WORKINGDIR+"/test_set/test_blend(private).csv")
    loader_test_private = DataLoader(
        ds_test_private, batch_size=config.BATCH_SIZE_CHECKACC, num_workers=2, pin_memory=True, shuffle=False
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    sys.stdout.write('\x1b[1A')
    sys.stdout.write('\x1b[2K')
    Slowprint("- Process Complete.")

    if config.LOAD_MODEL and config.CHECKPOINT_FILE_TRAIN_BLEND in os.listdir(config.SAVE_PATH):
        load_checkpoint(torch.load(config.SAVE_PATH + config.CHECKPOINT_FILE_TRAIN_BLEND),
                        model, optimizer, lr=1e-4)
        model.train()

    Previous_Kappa_Score = 0
    Current_Kappa_Score = 0
    HighestScorePublic, HighestScorePrivate = GetPreviousScore(
        config.WORKINGDIR+"/train_set/ScoreLog_TrainBlend.txt")
    DecreasetimesCount = 0
    EpochWithoutImprove = 0

    for _ in range(10):
        os.system("cls")

        losses = []
        Slowprint(f"- Epoch number {_ + 1}:")

        count = 0
        loop = tqdm(loader, desc="Training Model: ", leave=False, ascii=config.ASCII,
                    delay=config.DELAYTIME, colour=config.COLOR, bar_format=config.BARFORMAT)
        for x, y, files in loop:
            count += 1

            x = x.to(config.DEVICE).float()
            y = y.to(config.DEVICE).view(-1).float()

            # forward
            scores = model(x).view(-1)
            loss = loss_fn(scores, y)
            losses.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
            if int(loop.total) == count:
                loop.colour = config.COLOR_COMPLETE
                loop.update()

        loop.close()
        Slowprint(
            f" - Average loss over epoch: {sum(losses)/len(losses):.4f}")

        preds, labels = check_accuracy(loader_test_public, model)
        Current_Kappa_Score = cohen_kappa_score(
            labels, preds, weights='quadratic')
        Slowprint(
            f" - QuadraticWeightedKappa (Test Public): {Current_Kappa_Score:.4f}")
        if (HighestScorePublic < Current_Kappa_Score):
            HighestScorePublic = Current_Kappa_Score

        preds, labels = check_accuracy(loader_test_private, model)
        Current_Kappa_Score = cohen_kappa_score(
            labels, preds, weights='quadratic')
        Slowprint(
            f" - QuadraticWeightedKappa (Test Private): {Current_Kappa_Score:.4f}")

        if (HighestScorePrivate < Current_Kappa_Score):
            HighestScorePrivate = Current_Kappa_Score
        if (Current_Kappa_Score < Previous_Kappa_Score):
            Slowprint('\033[93m'+"- Test Score decreasing!"+"\033[0m")
            DecreasetimesCount += 1
            EpochWithoutImprove += 1
        else:
            if config.SAVE_MODEL and HighestScorePrivate <= Current_Kappa_Score:
                checkpoint = {"state_dict": model.state_dict(
                ), "optimizer": optimizer.state_dict()}
                save_checkpoint(
                    checkpoint, filename=config.SAVE_PATH + config.CHECKPOINT_FILE_TRAIN_BLEND)
                EpochWithoutImprove = 0
                ScoreLogFile(config.WORKINGDIR+"/train_set/ScoreLog_TrainBlend.txt",
                             HighestScorePublic, HighestScorePrivate)
            EpochWithoutImprove += 1
            DecreasetimesCount = 0
        Previous_Kappa_Score = Current_Kappa_Score

        if (DecreasetimesCount == 2 or EpochWithoutImprove == 3):
            Slowprint('\033[93m'+"- Model does not improve.""\033[0m")
            break

    Slowprint("Exiting...")
    sleep(1)
    os.system("cls")


def GetPredictions(CSV_FILE_PATH, SaveName):
    Slowprint("Setting up model and dataset...")
    model = MyModel().to(config.DEVICE)
    Predictions_ds = MyDataset(csv_file=CSV_FILE_PATH)
    Predictions_Loader = DataLoader(
        Predictions_ds, batch_size=config.BATCH_SIZE_CHECKACC, num_workers=2, pin_memory=True, shuffle=False
    )
    sys.stdout.write('\x1b[1A')
    sys.stdout.write('\x1b[2K')
    Slowprint("- Process Complete.")
    make_prediction(model=model, loader=Predictions_Loader, file=SaveName)


if __name__ == "__main__":
    os.system("cls")
    train.TrainModel()
    # train.GetDataForTrainBlend()
    # Train_blend()
