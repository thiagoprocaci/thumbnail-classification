import torch
from pyyoutube import Api

import skimage
from skimage import io, exposure
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import argparse
from random import shuffle

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from pathlib import Path


class YouTubeCrawler:

    PEIXE_BABEL = 'CanalPeixeBabel'
    NERDOLOGIA = 'nerdologia'
    #PIRULA = 'Pirulla25'

    def __init__(self):
        self.channels = [YouTubeCrawler.PEIXE_BABEL, YouTubeCrawler.NERDOLOGIA]
        try:
            api_key = ''
            self.api = Api(api_key=api_key)
        except:
            print("Ops.. Problem when creating an instance of YouTube API")

    def fetchThumbnails(self):
        if self.needsDownload() == False:
            print("You already downloaded the thumbnails")
            return
        thumbnailList = []

        nbins = 25
        for idx, channel in enumerate(self.channels):
            print('Visiting channel ' + channel)


            channelInfo = self.api.get_channel_info(channel_name=channel)
            playlistUploads = channelInfo.items[0].to_dict()['contentDetails']['relatedPlaylists']['uploads']

            playlistItens = self.api.get_playlist_items(playlist_id=playlistUploads, count=400)


            for k, item in enumerate(playlistItens.items):
                try:
                    ## Coletando informação da thumbnail de cada vídeo
                    videoId = item.snippet.resourceId.videoId
                    print('Getting thumbnail of video ' + videoId)
                    video = self.api.get_video_by_id(video_id=videoId)

                    thumbnailUrl = video.items[0].to_dict()['snippet']['thumbnails']['medium']['url']

                    ## Lendo a imagem e extraindo os histogramas
                    img = skimage.img_as_float(io.imread(thumbnailUrl))
                    histograms = [exposure.histogram(img[:, :, i], nbins=nbins, normalize=True)[0] for i in range(img.shape[-1])]

                    thumbnail = Thumbnail(thumbnailUrl, histograms, channel, idx)
                    thumbnailList.append(thumbnail)
                except:
                    print(k)
            np.savez_compressed('thumbnails', thumbnails=thumbnailList)


    def needsDownload(self):
        thumbnail_file = Path("thumbnails.npz")
        if thumbnail_file.is_file():
            return False
        return True



class Thumbnail:

    def __init__(self, url, rgbColors, classification, classificationNumber):
        self.url = url
        self.rgbColors = []

        red = np.array(rgbColors[0])
        green = np.array(rgbColors[1])
        blue = np.array(rgbColors[2])

        self.rgbColors.append(red)
        self.rgbColors.append(green)
        self.rgbColors.append(blue)
        self.rgbColors.append(red + green + blue)

        self.classification = classification
        self.classificationNumber = classificationNumber

    def plot(self):
        img = skimage.img_as_float(io.imread(self.url))
        plt.figure()
        plt.imshow(img)

        fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        axs[0].imshow(img[:, :, 0], cmap='Reds')
        axs[0].grid(True)
        axs[1].imshow(img[:, :, 1], cmap='Greens')
        axs[1].grid(True)
        axs[2].imshow(img[:, :, 2], cmap='Blues')
        axs[2].grid(True)

        nbins = 25
        histograms = [exposure.histogram(img[:, :, i], nbins=nbins, normalize=True)[0] for i in range(img.shape[-1])]
        fig, axs = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
        axs[0].bar(np.arange(len(histograms[0])), histograms[0], color='r')
        axs[0].axhline(0.03, 0, 50)
        axs[1].bar(np.arange(len(histograms[1])), histograms[1], color='g')
        axs[1].axhline(0.03, 0, 50)
        axs[2].bar(np.arange(len(histograms[2])), histograms[2], color='b')
        axs[2].axhline(0.03, 0, 50)

        plt.show()


    @staticmethod
    def loadThumbnails():
        thumbnails = np.load('thumbnails.npz', allow_pickle=True)
        thumbList = thumbnails['thumbnails'].tolist()
        thumbnailList = []
        for t in thumbList:
            if t.classification in (YouTubeCrawler.PEIXE_BABEL, YouTubeCrawler.NERDOLOGIA):
                thumbnailList.append(Thumbnail(t.url, t.rgbColors, t.classification, t.classificationNumber))
        shuffle(thumbnailList)
        return thumbnailList

    @staticmethod
    def getClassNameByNumber(classNumber):
        thumbnailList = Thumbnail.loadThumbnails()
        for t in thumbnailList:
            if t.classificationNumber == classNumber:
                return t.classification
        return None

class ThumbnailData(Dataset):

    def __init__(self, thumbnails):
        self.thumbnails = thumbnails


    def __len__(self):
        return len(self.thumbnails)

    def __getitem__(self, idx):
        return np.asarray(self.thumbnails[idx].rgbColors), np.asarray(self.thumbnails[idx].classificationNumber)



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 2, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1408, 400)
        self.fc2 = nn.Linear(400, 10)
        self.fc3 = nn.Linear(10, 2)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.dropout2(x)
        output = F.log_softmax(x)
        return output



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.unsqueeze(dim=1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
        if args.dry_run:
            break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    y_validation_list = []
    y_validation_pred_list = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.unsqueeze(dim=1)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            y_validation_pred = np.asarray(pred).ravel()
            y_validation = np.asarray(target.view_as(pred)).ravel()
            y_validation_list.extend(y_validation)
            y_validation_pred_list.extend(y_validation_pred)

    test_loss /= len(test_loader.dataset)
    precision_validation = precision_score(y_validation_list, y_validation_pred_list, average="macro")
    recall_validation = recall_score(y_validation_list, y_validation_pred_list, average="macro")
    f1_validation = f1_score(y_validation_list, y_validation_pred_list, average="macro")

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f} \n '.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset),
        precision_validation,
        recall_validation,
        f1_validation
    ))


def predict(thumbnail_url, model, device):
    img = skimage.img_as_float(io.imread(thumbnail_url))
    plt.figure()
    plt.imshow(img)

    histograms = [exposure.histogram(img[:, :, i], nbins=25, normalize=True)[0] for i in range(img.shape[-1])]

    thumbnail_to_pred = Thumbnail(thumbnail_url, histograms, None, None)
    data = np.asarray(thumbnail_to_pred.rgbColors)

    data = torch.Tensor(data).double().to(device)
    data = torch.unsqueeze(data, 0)
    data = torch.unsqueeze(data, 0)

    out = model(data)
    pred = out.argmax(dim=1, keepdim=True)
    plt.title(Thumbnail.getClassNameByNumber(pred), fontsize=16)
    plt.show()



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Thumbnail Example')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 10)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    crawler = YouTubeCrawler()
    crawler.fetchThumbnails()
    thumbnails = Thumbnail.loadThumbnails()
    #thumbnails[0].plot()
    data = ThumbnailData(thumbnails)

    validation_split = .15
    dataset_size = len(data)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    trainData = [data[index] for index in train_indices]
    testData = [data[index] for index in val_indices]

    train_loader = torch.utils.data.DataLoader(trainData,  **train_kwargs)
    test_loader = torch.utils.data.DataLoader(testData,  **test_kwargs)


    model = Net().to(device).double()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "thumbnail_cnn.pt")

    predict(thumbnails[0].url, model, device)

if __name__ == '__main__':
    main()
