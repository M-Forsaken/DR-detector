import sys
import os
from datetime import datetime
from time import sleep
import torch
import pandas as pd
import numpy as np
import config
from tqdm import tqdm
import torch.nn.functional as F


def check_accuracy(loader, model, device="cuda"):
    model.eval()
    all_preds, all_labels = [], []
    num_correct = 0
    num_samples = 0
    count = 0
    loop = tqdm(loader, desc="Checking Accuracy: ", leave=False, ascii=config.ASCII,
                delay=config.DELAYTIME, colour=config.COLOR, bar_format=config.BARFORMAT)
    for x, y, filename in loop:
        count += 1

        x = x.to(device=device)
        y = y.to(device=device)

        with torch.no_grad():
            predictions = model(x)

        # Convert MSE floats to integer predictions
        predictions[predictions < 0.5] = 0
        predictions[(predictions >= 0.5) & (predictions < 1.5)] = 1
        predictions[(predictions >= 1.5) & (predictions < 2.5)] = 2
        predictions[(predictions >= 2.5) & (predictions < 3.5)] = 3
        predictions[(predictions >= 3.5) & (predictions < 100)] = 4
        predictions = predictions.long().view(-1)
        y = y.view(-1)

        num_correct += (predictions == y).sum()
        num_samples += predictions.shape[0]

        # add to lists
        all_preds.append(predictions.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

        if int(loop.total) == count:
            loop.colour = config.COLOR_COMPLETE
            loop.update()
    loop.close()
    Slowprint(
        f"- Got {num_correct} / {num_samples} with accuracy of {float(num_correct) / float(num_samples) *100:.2f}%"
    )
    model.train()
    return np.concatenate(all_preds, axis=0, dtype=np.int64), np.concatenate(
        all_labels, axis=0, dtype=np.int64
    )


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    Slowprint("- AutoSaving...")
    torch.save(state, filename)
    sys.stdout.write('\x1b[1A')
    sys.stdout.write('\x1b[2K')
    Slowprint("- AutoSaved.")


def load_checkpoint(checkpoint, model, optimizer, lr):
    Slowprint("- Loading Model...")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    sys.stdout.write('\x1b[1A')
    sys.stdout.write('\x1b[2K')
    Slowprint("- Model Loaded.")

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def Slowprint(PrintObject, end="\n", PrintRate=5):
    for i in PrintObject:
        sleep(1/(PrintRate*10))
        print(i, end='')
    print(end=end)


def get_csv_for_blend(loader, model, output_csv_file):
    model.eval()
    filename_first = []
    filename_second = []
    labels_first = []
    labels_second = []
    all_features = []
    count = 0
    loop = tqdm(loader, desc="Getting data: ", leave=False, ascii=config.ASCII,
                delay=config.DELAYTIME, colour=config.COLOR, bar_format=config.BARFORMAT)

    for idx, (images, y, image_files) in enumerate(loop):
        count += 1

        images = images.to(config.DEVICE)

        with torch.no_grad():
            features = F.adaptive_avg_pool2d(
                model.extract_features(images), output_size=1
            )
            features_logits = features.reshape(
                features.shape[0] // 2, 2, features.shape[1])
            preds = model(images).reshape(images.shape[0] // 2, 2, 1)
            new_features = (
                torch.cat([features_logits, preds], dim=2)
                .view(preds.shape[0], -1)
                .cpu()
                .numpy()
            )
            all_features.append(new_features)
            filename_first += image_files[::2]
            filename_second += image_files[1::2]
            labels_first.append(y[::2].cpu().numpy())
            labels_second.append(y[1::2].cpu().numpy())
        if int(loop.total) == count:
            loop.colour = config.COLOR_COMPLETE
            loop.update()

    loop.close()
    all_features = np.concatenate(all_features, axis=0)
    df = pd.DataFrame(
        data=all_features, columns=[
            f"f_{idx}" for idx in range(all_features.shape[1])]
    )
    df["label_first"] = np.concatenate(labels_first, axis=0)
    df["label_second"] = np.concatenate(labels_second, axis=0)
    df["file_first"] = filename_first
    df["file_second"] = filename_second
    Slowprint("Writing data to csv file...")
    df.to_csv(output_csv_file, index=False)
    model.train()
    sys.stdout.write('\x1b[1A')
    sys.stdout.write('\x1b[2K')
    Slowprint("Process complete.")


def ScoreLogFile(Filename, HighestScorePublic, HighestScorePrivate):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    File = open(f"{Filename}", "a")
    File.write("- Train completion date: "+dt_string+"\n")
    File.write(f" HighestScorePublic: {HighestScorePublic:.4f}"+"\n")
    File.write(f" HighestScorePrivate: {HighestScorePrivate:.4f}"+"\n")
    File.write("\n")
    File.close()


def GetPreviousScore(Filename):
    """
    This Function Return: 
        >>> HighestScorePublic 
        >>> HighestScorePrivate    
    """
    if not os.path.exists(Filename):
        return (0, 0)
    File = open(f"{Filename}", "r")
    latest = ""
    label = "=> Train completion date: "
    for i in File:
        if (label in i):
            time = i.replace(label, "")
            if (latest < time):
                latest = time
    File.seek(0)
    for i in File:
        if (latest in i):
            HighestScorePublic = float(File.__next__().split()[1])
            HighestScorePrivate = float(File.__next__().split()[1])
    File.close()
    return HighestScorePublic, HighestScorePrivate
