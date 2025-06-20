import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import argparse
from data import MNISTDataset
from model import SimpleCNN
import pandas as pd
from torchvision.transforms import v2
from tqdm import tqdm
import json

# print(torch.cuda.is_available())


# DEFINICION DE DATASETS TRAINING Y VALIDATION
# DATALOADERS


def main(args):
    # DATASET DEFINITION
    transforms = v2.Compose([v2.ToTensor()])


    training_images_folder = os.path.join(args.dataset_path, 'train')
    training_df_path = os.path.join(args.dataset_path, 'training_labels.csv')
    test_images_folder = os.path.join(args.dataset_path, 'test')
    test_df_path = os.path.join(args.dataset_path, 'test_labels.csv')
    training_df = pd.read_csv(training_df_path)
    training_df = training_df.sample(frac=0.01).reset_index(drop=True)
    validation_df = training_df.loc[0:int(0.2*len(training_df)) :].reset_index(drop=True)
    training_df = training_df.loc[int(0.2*len(training_df))::, :].reset_index(drop=True)

    test_df = pd.read_csv(test_df_path)

    training_dataset = MNISTDataset(training_images_folder, training_df, transforms=transforms)
    validation_dataset = MNISTDataset(training_images_folder, validation_df, transforms=transforms)
    test_dataset = MNISTDataset(test_images_folder, test_df, transforms=transforms)
    
    training_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # CRITERION, OPTIMIZADOR
    device = torch.device('cpu')
    model = SimpleCNN(10, 1, 28)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print('Everything has been initialized')


    metrics = {'train_loss' : [], 'train_acc' : [], 'val_loss' : [], 'val_acc' : []}
    for epoch_index in range(args.nb_epochs):
        running_loss, running_corrects = 0.0, 0
        model.train()
        print(f'Starting epoch {epoch_index}')
        for batch_x, batch_y in tqdm(training_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            predictions = torch.max(outputs, 1)[1]
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            running_corrects += torch.sum(predictions == batch_y.data)

        metrics['train_loss'] += [running_loss / len(training_dataset)]
        metrics['train_acc'] += [(running_corrects.double() / len(training_dataset)).item()]
        print(f'Epoch train loss: {metrics["train_loss"][-1]}')
        print(f'Epoch train acc: {metrics["train_acc"][-1]}')

        running_loss, running_corrects = 0.0, 0

        model.eval()
        for batch_x, batch_y in validation_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, batch_y)
            running_loss += loss.item() * batch_x.size(0)
            running_corrects += torch.sum(preds == batch_y.data)
        metrics['val_loss'] += [running_loss / len(validation_dataset)]
        metrics['val_acc'] += [(running_corrects.double() / len(validation_dataset)).item()]
        print(f'Epoch val loss: {metrics["val_loss"][-1]}')
        print(f'Epoch val acc: {metrics["val_acc"][-1]}')

    os.makedirs(args.saving_folder, exist_ok=True)
    with open(os.path.join(args.saving_folder, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f)
    ## EVALUACION EN TEST
# FOR DE ENTRENAMIENTO
# EVALUACION DEL MODELO EN TEST

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--nb-epochs', type=int, default=10)
    parser.add_argument('--saving-folder', type=str, required=True)
    args = parser.parse_args()

    main(args)

