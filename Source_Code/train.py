from time import sleep
import torch
from torch import nn, optim
import os
import config
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from efficientnet_pytorch import EfficientNet
from dataset import DRDataset
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    Slowprint,
    get_csv_for_blend,
    ScoreLogFile,
    GetPreviousScore,
)


def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    losses = []
    count = 0
    loop = tqdm(loader, desc="Training Model: ",
                delay=config.DELAYTIME, leave=False, ascii=config.ASCII, colour=config.COLOR, bar_format=config.BARFORMAT)
    for data, targets, _ in loop:
        count += 1

        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores, targets.unsqueeze(1).float())

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())

        if int(loop.total) == count:
            loop.colour = config.COLOR_COMPLETE
            loop.update()

    loop.close()
    Slowprint("- Training Complete.")
    Slowprint(f"- Average loss over epoch: {sum(losses)/len(losses):.3f}")
    del losses


def InitTrainData():
    train_ds = DRDataset(
        images_folder=config.WORKINGDIR+"/train_set/images_preprocessed_1000/",
        path_to_csv=config.WORKINGDIR+"/train_set/trainLabels.csv",
        transform=config.train_transforms,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE_TRAIN,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
    )
    return train_loader


def InitTestData():
    test_ds_public = DRDataset(
        images_folder=config.WORKINGDIR+"/test_set/images_preprocessed_1000/",
        path_to_csv=config.WORKINGDIR+"/test_set/test_public.csv",
        transform=config.val_transforms,
    )
    test_ds_private = DRDataset(
        images_folder=config.WORKINGDIR+"/test_set/images_preprocessed_1000/",
        path_to_csv=config.WORKINGDIR+"/test_set/test_private.csv",
        transform=config.val_transforms,
    )
    test_loader_private = DataLoader(
        test_ds_private,
        batch_size=config.BATCH_SIZE_CHECKACC,
        num_workers=2,
    )
    test_loader_public = DataLoader(
        test_ds_public,
        batch_size=config.BATCH_SIZE_CHECKACC,
        num_workers=2,
    )
    return test_loader_public, test_loader_private


def TrainModel():
    loss_fn = nn.MSELoss()

    model = EfficientNet.from_pretrained("efficientnet-b3")
    # model._fc = nn.Linear(1792, 1) # layers for B4 model
    model._fc = nn.Linear(1536, 1)  # layers for B3 model
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL and config.CHECKPOINT_FILE_TRAIN in os.listdir(config.SAVE_PATH):
        load_checkpoint(torch.load(config.SAVE_PATH + config.CHECKPOINT_FILE_TRAIN),
                        model, optimizer, config.LEARNING_RATE)

    Previous_Kappa_Score = 0
    Current_Kappa_Score = 0
    DecreasetimesCount = 0
    HighestScorePublic, HighestScorePrivate = GetPreviousScore(
        config.WORKINGDIR+"/train_set/ScoreLog_Train.txt")
    EpochWithoutImprove = 0

    for epoch in range(config.NUM_EPOCHS):
        train_loader = InitTrainData()
        train_one_epoch(train_loader, model, optimizer,
                        loss_fn, scaler, config.DEVICE)
        del train_loader
        torch.cuda.empty_cache()
        test_loader_public, test_loader_private = InitTestData()
        preds, labels = check_accuracy(test_loader_public, model)
        Current_Kappa_Score = cohen_kappa_score(
            labels, preds, weights='quadratic')
        Slowprint(
            f" +) QuadraticWeightedKappa (Test Public): {Current_Kappa_Score:.4f}")
        if (HighestScorePublic < Current_Kappa_Score):
            HighestScorePublic = Current_Kappa_Score
        preds, labels = check_accuracy(test_loader_private, model)
        Current_Kappa_Score = cohen_kappa_score(
            labels, preds, weights='quadratic')
        Slowprint(
            f" +) QuadraticWeightedKappa (Test Private): {Current_Kappa_Score:.4f}")
        if (HighestScorePrivate < Current_Kappa_Score):
            HighestScorePrivate = Current_Kappa_Score
        del (test_loader_private, test_loader_public)
        torch.cuda.empty_cache()
        if (Current_Kappa_Score < Previous_Kappa_Score):
            Slowprint('\033[93m'+"- Test Score decreasing!"+"\033[0m")
            DecreasetimesCount += 1
            EpochWithoutImprove += 1
        else:
            if config.SAVE_MODEL and HighestScorePrivate <= Current_Kappa_Score:
                checkpoint = {"state_dict": model.state_dict(
                ), "optimizer": optimizer.state_dict()}
                save_checkpoint(
                    checkpoint, filename=config.SAVE_PATH + config.CHECKPOINT_FILE_TRAIN)
                EpochWithoutImprove = 0
                ScoreLogFile(config.WORKINGDIR+"/train_set/ScoreLog_Train.txt",
                             HighestScorePublic, HighestScorePrivate)
            EpochWithoutImprove += 1
            DecreasetimesCount = 0
        Previous_Kappa_Score = Current_Kappa_Score

        if (DecreasetimesCount == 2 or EpochWithoutImprove == 3):
            Slowprint('\033[93m'+"- Model does not improve.""\033[0m")
            break

        if config.SAVE_MODEL :
            checkpoint = {"state_dict": model.state_dict(
            ), "optimizer": optimizer.state_dict()}
            save_checkpoint(
                checkpoint, filename=config.SAVE_PATH + config.CHECKPOINT_FILE_TRAIN)
            EpochWithoutImprove = 0
            ScoreLogFile(config.WORKINGDIR+"/train_set/ScoreLog_Train.txt",
                        HighestScorePublic, HighestScorePrivate)

    Slowprint("Exiting...")
    sleep(1)
    os.system("cls")


def GetDataForTrainBlend():
    train_ds = DRDataset(
        images_folder=config.WORKINGDIR+"/train_set/images_preprocessed_1000/",
        path_to_csv=config.WORKINGDIR+"/train_set/trainLabels.csv",
        transform=config.val_transforms,
    )
    test_ds_public = DRDataset(
        images_folder=config.WORKINGDIR+"/test_set/images_preprocessed_1000/",
        path_to_csv=config.WORKINGDIR+"/test_set/test_public.csv",
        transform=config.val_transforms,
    )
    test_ds_private = DRDataset(
        images_folder=config.WORKINGDIR+"/test_set/images_preprocessed_1000/",
        path_to_csv=config.WORKINGDIR+"/test_set/test_private.csv",
        transform=config.val_transforms,
    )
    test_loader_private = DataLoader(
        test_ds_private,
        batch_size=config.BATCH_SIZE_CHECKACC,
        num_workers=2,
        shuffle=False,
    )
    test_loader_public = DataLoader(
        test_ds_public,
        batch_size=config.BATCH_SIZE_CHECKACC,
        num_workers=2,
        shuffle=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE_CHECKACC,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
    )

    model = EfficientNet.from_pretrained("efficientnet-b3")
    model._fc = nn.Linear(1536, 1)  # layers for B3 model
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    if config.LOAD_MODEL and config.CHECKPOINT_FILE in os.listdir():
        load_checkpoint(torch.load(config.CHECKPOINT_FILE),
                        model, optimizer, config.LEARNING_RATE)

    # Run after training is done and you've achieved good result
    # on validation set, then run train_blend.py file to use information
    # about both eyes concatenated
    Slowprint("Getting test_blend(public).csv")
    get_csv_for_blend(test_loader_public, model,
                      config.WORKINGDIR+"/test_set/test_blend(public).csv")

    Slowprint("Getting test_blend(private).csv")
    get_csv_for_blend(test_loader_private, model,
                      config.WORKINGDIR+"/test_set/test_blend(private).csv")

    Slowprint("Getting train_blend.csv")
    get_csv_for_blend(train_loader, model,
                      config.WORKINGDIR+"/train_set/train_blend.csv")

    Slowprint("Exiting...")
    sleep(1)
    os.system("cls")
