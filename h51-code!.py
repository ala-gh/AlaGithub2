#in hamon file 49 hast .
#mikham kami taghirat bedam. dataloader ha ro biyaram biron . on list avaliye ro bardaram. va ...

import pickle
from torch.utils.data.dataset import Dataset
from hazm import *
import csv
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

print('start')
#*********************************dataset class ****************************************
class custom_dataset(Dataset):

    def __init__(self, dataset_path,vocab_path):

        getdata = open(dataset_path, "r", encoding='utf-8')
        name = getdata.name
        print('data:',name)
        getvocab = csv.reader(open(vocab_path, encoding='utf-8'))
        data = getdata.read()
        self.data_split = data.split('\n')  # split samples


        self.vocabulary = {}
        self.sample =''
        for row in getvocab:

            if not row:
                1 + 1
            else:
                value = row[0]
                key = row[1]
                if key in self.vocabulary:
                    pass
                self.vocabulary[value] = key

        self.a_file = open("dicVocabulary.pkl", "wb")
        pickle.dump(self.vocabulary, self.a_file)
        self.a_file.close()

        self.labels = {' Literature': '0', ' Political': '1', ' Religious': '2', ' Scientific': '3', ' Social': '4',
                       ' Sports': '5', }


# ***************************return indexvector, label, length of each sample******************************************

    def indexLabel(self,sample, vocabulary, labels):
        splitSample = sample.split(',')
        textSample = splitSample[0]
        labelSample = splitSample[1]

        indexVector = []
        indexlabel = 1

        sampleTokens = word_tokenize(textSample)
        for token in sampleTokens:
            if token in vocabulary:
                indexVector.append(vocabulary[token])
            else:
                indexVector.append("unk")


        if labelSample in labels:
            indexlabel = labels[labelSample]

        lengthSample = len(textSample)

        indexVector = np.array(indexVector)
        IndexVector_withoutUNK = np.delete(indexVector, np.where(indexVector == 'unk')[0], axis=0)
        lengthSample = len(IndexVector_withoutUNK)
        IndexVector_withoutUNK = IndexVector_withoutUNK.astype(np.int)
        IndexVector_withoutUNK = torch.LongTensor(IndexVector_withoutUNK)
        # print(label)
        indexlabel = int(indexlabel)

        return (IndexVector_withoutUNK, indexlabel, lengthSample, textSample)

    def __len__(self):
        return len(self.data_split)

    def __getitem__(self, idx):
        self.sample = self.data_split[idx]
        self.sampleIndexVector, self.sampleIndexlabel, \
        self.samplelengthSample, self.sampleText = self.indexLabel(self.sample,self.vocabulary,self.labels)

        return self.sampleIndexVector, self.sampleIndexlabel, self.samplelengthSample, self.sampleText

#**********************************object of data***************************************
data = custom_dataset('./newAsoSoftMiniLabel.txt', './outputnewAsoSoftSmall.csv')


train_dataset = []
for vector, label, length, context in data:
    train_dataset.append([vector, label, length,context])

#*********************** split dataset *************************************************
train_percent = 0.95
validation_percent = 0.05
# print('percent of train, valid:',train_percent,'and',validation_percent)
train_len = int(len(train_dataset) * train_percent)
sub_train_, sub_valid_ = random_split\
    (train_dataset, [train_len, len(train_dataset) - train_len])

#***********************************RNN class ******************************************
class RNN(nn.Module):
    def __init__(self, embedding_dim, num_vocabs, num_layer, hidden_dim, num_classes, batch_size):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.features = num_vocabs
        self.batch_size = batch_size

        self.embedding = nn.Embedding(num_vocabs, embedding_dim)
        self.rnn = nn.RNN(embedding_dim,hidden_dim)                       # num_layers=num_layer
        # self.lstm = nn.LSTM(embedding_dim,hidden_dim,1)
        # self.gru = nn.GRU(embedding_dim,hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, text):

        embedded = self.embedding(text)
        a=text.size(1)
        b= embedded.size(1)
        hidden_zero = torch.zeros(1, text.size(1), self.hidden_dim).to(device)
        cell_zero = torch.zeros(1, text.size(1), self.hidden_dim).to(device)

        # output,( hidden,cell) = self.lstm(embedded,(hidden_zero,cell_zero))   #,self.hidden
        output, hidden = self.rnn(embedded, hidden_zero)   #,self.hidden

        b2 = output[-1, :, :]
        b4 = hidden.squeeze(0)

        assert torch.equal(output[-1, :, :],hidden.squeeze(0))       #hidden.squeeze(0)
        return self.fc(hidden.squeeze(0))


class RNN2(nn.Module):
    def __init__(self, embedding_dim, num_vocabs, num_layer, hidden_dim, num_classes, batch_size):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.features = num_vocabs
        self.batch_size = batch_size


        self.embedding = nn.Embedding(num_vocabs, embedding_dim)
        # self.rnn = nn.RNN(embedding_dim,hidden_dim)                       # num_layers=num_layer
        # self.lstm = nn.LSTM(embedding_dim,hidden_dim,1)
        self.gru = nn.GRU(embedding_dim,hidden_dim,1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, text, offset, label):

        embedded = self.embedding(text)
        context1 = []

        for i in range(len(label)):
            a = offset[i].tolist()
            b = offset[i + 1].tolist()
            context1.append(embedded[a:b])

        paded = pad_sequence(context1)

        hidden_zero = torch.zeros(1, paded.size(1), self.hidden_dim).to(device)
        cell_zero = torch.zeros(1, paded.size(1), self.hidden_dim).to(device)


        # output,( hidden,cell) = self.lstm(paded,(hidden_zero,cell_zero))   #,self.hidden
        output, hidden = self.gru(paded, hidden_zero)   #,self.hidden


        b1 = output
        b2 = output[-1, :, :]
        b3 = hidden
        b4 = hidden.squeeze(0)

        assert torch.equal(output[-1, :, :],hidden.squeeze(0))       #hidden.squeeze(0)
        return self.fc(hidden.squeeze(0))

#**********************************object of RNN******************************************
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = len(data.vocabulary)
EMBEDDING_DIM = 512
HIDDEN_DIM = 128
OUTPUT_DIM = 6
BATCH_SIZE = 1
N_EPOCHS = 5
NUM_LAYER = 4
lr=1e-3
min_valid_loss = float('inf')

model = RNN(EMBEDDING_DIM,INPUT_DIM, NUM_LAYER, HIDDEN_DIM, OUTPUT_DIM, BATCH_SIZE)
model2 = RNN2(EMBEDDING_DIM,INPUT_DIM, NUM_LAYER, HIDDEN_DIM, OUTPUT_DIM, BATCH_SIZE)

#****************************parameters************************************************

import torch.optim as optim


# optimizer = optim.SGD(model.parameters(), lr)
# criterion = torch.nn.CrossEntropyLoss().to(device)
#
# model = model.to(device)
# criterion = criterion.to(device)
def optimizer_criterion(model):
    optimizer = optim.SGD(model.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model = model.to(device)
    criterion = criterion.to(device)
    return optimizer,criterion,model
optimizer,criterion,model = optimizer_criterion(model)
# ***************************** generate batch *****************************************

def generate_batch(batch):
    label = torch.tensor([entry[1] for entry in batch])
    text = [entry[0] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]

    offsets = torch.tensor(offsets).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label

def generate_batch2(batch):
    label = torch.tensor([entry[1] for entry in batch])
    text =  [entry[0] for entry in batch]
    # offsets = [0] + [len(entry) for entry in text]

    # offsets = torch.tensor(offsets).cumsum(dim=0)
    # text = torch.cat(text)
    return text, label

train_batches = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=generate_batch)
valid_batches = DataLoader(sub_valid_, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=generate_batch)
#**************************** train function  *******************************************
def train(model, optimizer, criterion, accu, N_EPOCHS):
    if N_EPOCHS==0:
        print('class name :',model.__class__.__name__)
        print('function name:',train.__name__)
        print('function name:', accu.__name__)
        print(model)
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in sub_train_:
        textvector= batch[1]
        label= torch.tensor([batch [0]])
        textvector, label= textvector.to(device), label.to(device)

        optimizer.zero_grad()
        textvector = textvector.unsqueeze(1)

        pred = model(textvector)
        predictions = pred.squeeze(1)
        loss = criterion(predictions, label)

        acc = accu(predictions,label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(sub_train_), epoch_acc / len(sub_train_)

# -----------------------------------------------------------------------------------------------------------

def train2(model, optimizer, criterion,accu,N_EPOCHS):
    if N_EPOCHS==0:
        print('class name :', model.__class__.__name__)
        print('function name:', train2.__name__)
        print('function name:', accu.__name__)
        print(model)
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for i,batch in enumerate(train_batches):

        textvector = batch[0]
        offsets = batch[1]
        label = batch[2]
        textvector, offsets,label= textvector.to(device), offsets.to(device), label.to(device)

        optimizer.zero_grad()
        pred = model(textvector, offsets, label)
        predictions = pred.squeeze(1)
        loss = criterion(predictions, label)
        acc = accu(predictions, label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        # x,y = epoch_acc, len(batchs)
        # print(x, y)
    return epoch_loss / len(train_batches), epoch_acc / len(train_batches)


#**************************** validation function  *******************************************
def validation2(model, criterion, accu, N_EPOCHS):
    if N_EPOCHS==0:
        print('class name :', model.__class__.__name__)
        print('function name:', validation2.__name__)
        print('function name:', accu.__name__)
        print(model)
    epoch_loss = 0
    epoch_acc = 0


    model.eval()
    with torch.no_grad():
        for batch in valid_batches:
            textvector = batch[0]
            offsets = batch[1]
            label = batch[2]
            textvector, offsets,label= textvector.to(device), offsets.to(device), label.to(device)

            pred = model(textvector, offsets, label)
            predictions = pred.squeeze(1)
            loss = criterion(predictions, label)
            acc = accu(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            # x,y = epoch_acc, len(batchs)
            # print(x, y)
    return epoch_loss / len(valid_batches), epoch_acc / len(valid_batches)

# ***************************  def accuracy ****************************************8**
def accu(predictions,label):
    predictions = torch.sigmoid(predictions)
    max = predictions.argmax()
    rounded_preds = max
    correct = (rounded_preds == label).float()
    acc = correct.sum() / len(correct)
    return acc


def accu2(predictions,label):
    _, predictions_max = torch.max(predictions, 1)
    correct = (predictions_max == label).float()
    acc = correct.sum() / len(correct)
    return acc

def accu3(predictions,label):
    max = predictions.argmax()
    correct = (max == label).float()
    acc = correct.sum() / len(correct)
    return acc

def accu4(predictions,label):

    newlabels = []
    for i in predictions:
        max = i.argmax()
        newlabels.append(int(max))
    c=[2]
    for i, j in enumerate(newlabels) :
        if j == label[i]:
           c= c.append(1)
        else:
            c=c.append(5)

    # correct = (newlabels == label).float()
    correct = c
    acc = correct.sum() / len(correct)-1
    return acc


#**************************time function **********************************************
import time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

#****************************** main ***************************************************
for epoch in range(N_EPOCHS):

    start_time = time.time()
    optimizer, criterion, model2 = optimizer_criterion(model2)
    train_loss, train_acc = train2(model2, optimizer, criterion,accu3,epoch)
    valid_loss, valid_acc = validation2(model2, criterion, accu3, epoch)
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\tvalid Loss: {valid_loss:.3f} | valid Acc: {valid_acc * 100:.2f}%')

PATH = 'savemodel-train2-model2-minidata.pth'
torch.save(model.state_dict(), PATH)





#
# batchs = DataLoader(sub_train_, batch_size=64, shuffle=True,
#                         collate_fn=generate_batch)
#
# next(iter(batchs))
# 35*16+1035*16+10

 # train_loss, train_acc = train(model, sub_train_, optimizer, criterion)
 #    #valid_loss, valid_acc = evaluate(model, sub_valid_, criterion)
#
# print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
# print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
# # print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

