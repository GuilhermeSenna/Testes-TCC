import os
import time
from tqdm import tqdm
import torch
import pickle
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,DataLoader

print(torch.__version__) # This code has been updated for PyTorch 1.0.0

# trainFeats, trainLabel = None, None

savePath = 'lecture5_output/'
# if not os.path.isdir(savePath):
#     os.makedirs(savePath)
#
#     # Loading the saved features
#     with open("trainFeats.pckl", "rb") as f:
#         trainFeats = pickle.load(f)
#         print(trainFeats)
#     with open("trainLabel.pckl", "rb") as f:
#         trainLabel = pickle.load(f)
#
#     with open("testFeats.pckl", "rb") as f:
#         testFeats = pickle.load(f)
#     with open("testLabel.pckl", "rb") as f:
#         testLabel = pickle.load(f)
#
#     # Generating 1-hot label vectors
#     trainLabel2 = np.zeros((50000, 10))
#     testLabel2 = np.zeros((10000, 10))
#     for d1 in range(trainLabel.shape[0]):
#         trainLabel2[d1, trainLabel[d1]] = 1
#     for d2 in range(testLabel.shape[0]):
#         testLabel2[d2, testLabel[d2]] = 1
#
#     print('Finished loading saved feature matrices from the disk!')

with open("trainFeats.pckl", "rb") as f:
    trainFeats = pickle.load(f)
    print(trainFeats)
with open("trainLabel.pckl", "rb") as f:
    trainLabel = pickle.load(f)

with open("testFeats.pckl", "rb") as f:
    testFeats = pickle.load(f)
with open("testLabel.pckl", "rb") as f:
    testLabel = pickle.load(f)

# Generating 1-hot label vectors
trainLabel2 = np.zeros((50000, 10))
testLabel2 = np.zeros((10000, 10))
for d1 in range(trainLabel.shape[0]):
    trainLabel2[d1, trainLabel[d1]] = 1
for d2 in range(testLabel.shape[0]):
    testLabel2[d2, testLabel[d2]] = 1

device = "cpu"
pinMem = False


# Defining the perceptron
class perceptron(nn.Module):
    def __init__(self,n_channels): #n_channels => length of feature vector
        super(perceptron, self).__init__()
        self.L = nn.Linear(n_channels,10) #Mapping from input to output
    def forward(self,x): #x => Input
        x = self.L(x) #Feed-forward
        x = F.softmax(x,dim=1) #Softmax non-linearity, dim=1 corresponds to labels
        return x

# Checking availability of GPU
# use_gpu = torch.cuda.is_available()
# if use_gpu:
#     print('GPU is available!')
#     device = "cuda"
#     pinMem = True
# else:
#     print('GPU is not available!')


# Creating pytorch dataset from the feature matices
trainDataset = TensorDataset(torch.from_numpy(trainFeats), torch.from_numpy(trainLabel2))
testDataset = TensorDataset(torch.from_numpy(testFeats), torch.from_numpy(testLabel2))
# Creating dataloader
trainLoader = DataLoader(trainDataset, batch_size=1, shuffle=True,num_workers=4, pin_memory=pinMem)
testLoader = DataLoader(testDataset, batch_size=1, shuffle=False,num_workers=4, pin_memory=pinMem)


# Definining the training routine
def train_model(model, criterion, num_epochs, learning_rate):
    start = time.time()
    train_loss = []  # List for saving the loss per epoch
    train_acc = []  # List for saving the accuracy per epoch
    tempLabels = []  # List for saving shuffled labels as fed into the network
    for epoch in range(num_epochs):
        epochStartTime = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        running_loss = 0.0
        # Loading data in batches
        batch = 0
        for data in tqdm(trainLoader):
            inputs, labels = data

            inputs, labels = inputs.float().to(device), labels.float().to(device)

            # Initializing model gradients to zero
            model.zero_grad()
            # Data feed-forward through the network
            outputs = model(inputs)
            # Predicted class is the one with maximum probability
            _, preds = outputs.data.max(1)
            # Finding the MSE
            loss = criterion(outputs, labels)
            # Accumulating the loss for each batch
            running_loss += loss.item()

            # Backpropaging the error
            if batch == 0:
                totalLoss = loss
                totalPreds = preds
                tempLabels = labels.data.cpu()
                batch += 1
            else:
                totalLoss += loss
                totalPreds = torch.cat((totalPreds, preds), 0)
                tempLabels = torch.cat((tempLabels, labels.data.cpu()), 0)
                batch += 1

        totalLoss = totalLoss / batch
        totalLoss.backward()

        # Updating the model parameters
        for f in model.parameters():
            f.data.sub_(f.grad.data * learning_rate)

        epoch_loss = running_loss / 50000  # Total loss for one epoch
        train_loss.append(epoch_loss)  # Saving the loss over epochs for plotting the graph

        # Accuracy per epoch
        tempLabels = tempLabels.numpy()
        _, totalLabels = np.where(tempLabels == 1)
        epoch_acc = np.sum(np.equal(totalPreds.cpu().numpy(), np.array(totalLabels))) / 50000.0
        train_acc.append(epoch_acc * 100)  # Saving the accuracy over epochs for plotting the graph

        epochTimeEnd = time.time() - epochStartTime
        print('Average epoch loss: {:.6f}'.format(epoch_loss))
        print('Average epoch accuracy: {:.4f} %'.format(epoch_acc * 100))
        print('-' * 25)
        # Plotting Loss vs Epochs
        fig1 = plt.figure(1)
        plt.plot(range(epoch + 1), train_loss, 'r--', label='train')
        if epoch == 0:
            plt.legend(loc='upper right')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Plot of training loss vs epochs')
        fig1.savefig(savePath + 'lossPlot.png')
        # Plotting Accuracy vs Epochs
        fig2 = plt.figure(2)
        plt.plot(range(epoch + 1), train_acc, 'g--', label='train')
        if epoch == 0:
            plt.legend(loc='upper left')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('Plot of training accuracy vs epochs')
        fig2.savefig(savePath + 'accPlot.png')

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model


featLength = 2+5+2
# Initilaizing the model
model = perceptron(featLength).to(device)
criterion = nn.MSELoss()
model = train_model(model,criterion,num_epochs=100,learning_rate=1) # Training the model

# Finding testing accuracy
test_running_corr = 0
# Loading data in batches
batches = 0
testLabels = []

model.eval()  # Testing the model in evaluation mode

for tsData in tqdm(testLoader):
    inputs, labels = tsData

    inputs, labels = inputs.float().to(device), labels.float()

    with torch.no_grad():  # No back-propagation during testing; gradient computation is not required

        # Feedforward train data batch through model
        output = model(inputs)
        # Predicted class is the one with maximum probability
        _, preds = output.data.max(1)
        if batches == 0:
            totalPreds = preds
            testLabels = torch.argmax(labels, dim=1)  # Converting 1-hot vector labels to integer labels
            batches = 1
        else:
            totalPreds = torch.cat((totalPreds, preds), 0)
            testLabels = torch.cat((testLabels, torch.argmax(labels, dim=1)), 0)

        # Finding total number of correct predictions
ts_corr = np.sum(np.equal(totalPreds.cpu().numpy(), testLabels.numpy()))
# Calculating accuracy
ts_acc = ts_corr / testLabels.shape[0]
print('Accuracy on test set = ' + str(ts_acc * 100) + '%')