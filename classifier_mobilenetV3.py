import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import sys
from sklearn.metrics import accuracy_score, classification_report

from tqdm import tqdm
import time
import warnings

warnings.simplefilter("ignore")


def images_transforms():
    data_transformation = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])
    return data_transformation


def get_classifier_model():
    m = torchvision.models.mobilenet_v3_large(pretrained=True)
    num_features = m.classifier[0].in_features
    m.classifier = nn.Sequential(
        nn.Linear(in_features=num_features, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=num_classes, bias=True)
    )

    m = m.to(device)
    return m


def test(m, loader):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        y_pred = []
        y_actual = []
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = m(images)

            y_actual += list(np.array(labels.detach().to('cpu')).flatten())
            # value ,index
            _, predicts = torch.max(outputs, 1)
            y_pred += list(np.array(predicts.detach().to('cpu')).flatten())
            # number of samples in current batch
            n_samples += labels.shape[0]

            n_correct += (predicts == labels).sum().item()

        y_actual = np.array(y_actual).flatten()
        y_pred = np.array(y_pred).flatten()
        # print(np.unique(y_pred))

        # acc = classification_report(y_actual,y_pred)
        # print(f"{acc}")

        acc = accuracy_score(y_actual, y_pred)
        print(f"Accuracy: {acc}")

    return acc


def train(m, train_loader, criterion, optimizer, val_loader, n_epochs=25):
    train_losses = []
    val_losses = []
    train_auc = []
    val_auc = []
    train_auc_epoch = []
    val_auc_epoch = []
    best_acc = 0.0
    min_loss = np.Inf

    since = time.time()
    for e in range(n_epochs):
        y_actual = []
        y_pred = []
        train_loss = 0.0
        val_loss = 0.0

        # Train the model
        m.train()
        for i, (images, labels) in enumerate(tqdm(train_loader, total=int(len(train_loader)))):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = m(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Loss and accuracy
            train_loss += loss.item()

            _, predicts = torch.max(outputs, 1)
            y_actual += list(labels.data.cpu().numpy().flatten())
            y_pred += list(predicts.detach().cpu().numpy().flatten())
        train_auc.append(accuracy_score(y_actual, y_pred))

        # Evaluate the model
        m.eval()
        for i, (images, labels) in enumerate(tqdm(val_loader, total=int(len(val_loader)))):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = m(images)
            loss = criterion(outputs, labels)

            # Loss and accuracy
            val_loss += loss.item()
            _, predicts = torch.max(outputs, 1)
            y_actual += list(labels.data.cpu().numpy().flatten())
            y_pred += list(predicts.detach().cpu().numpy().flatten())

        val_auc.append(accuracy_score(y_actual, y_pred))

        # Average losses and accuracies
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        training_auc = train_auc[-1]
        validation_auc = val_auc[-1]
        train_auc_epoch.append(training_auc)
        val_auc_epoch.append(validation_auc)

        # Updating the best validation accuracy
        if best_acc < validation_auc:
            best_acc = validation_auc

        # Saving best model
        if min_loss >= val_loss:
            torch.save(m.state_dict(), './models/best_model_512.pt')
            min_loss = val_loss

        print(
            'EPOCH {}/{} Train loss: {:.6f},Validation loss: {:.6f}, Train AUC: {:.4f}  Validation AUC: {:.4f}\n  '.format(
                e + 1, epochs, train_loss, val_loss, training_auc, validation_auc))
        print('-' * 10)
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation accuracy: {:4f}'.format(best_acc))
    return train_losses, val_losses, train_auc, val_auc, train_auc_epoch, val_auc_epoch


if __name__ == '__main__':
    load_path = sys.argv[1]

    # set hyperparameter
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMAGE_SIZE = (512, 512)
    batch_size = 10
    learning_rate = 0.0008
    epochs = 50
    num_classes = 4

    # get classifier model
    model = get_classifier_model()
    model.load_state_dict(torch.load('./models/best_model_512.pt'))

    # get dataset
    dataset = datasets.ImageFolder(load_path, transform=images_transforms())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # test result
    test(model, data_loader)

    # # code for training
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
    #
    # prepare data
    # train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    # test_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    #
    # # train
    # train_losses, val_losses, train_auc, val_auc, train_auc_epoch, val_auc_epoch = train(model, data_loader,
    #                                                                                      criterion, optimizer,
    #                                                                                      val_loader, epochs)
