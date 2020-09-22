from torch.utils.data.dataset import Dataset
from hazm import *
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from torch.utils.data.dataset import random_split
import numpy as np
import random

print("start")
#******************************** get data ***************************
class custom_dataset(Dataset):

    def __init__(self, dataset_path,vocab_path):

        getdata = open(dataset_path, "r", encoding='utf-8')
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

        # ***************************return indexvector, label, length of each sample***********

        self.labels = {' Literature': '0', ' Political': '1', ' Religious': '2', ' Scientific': '3', ' Social': '4',
                  ' Sports': '5',}

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

data = custom_dataset('./AsoSoftLabel.txt', './output3.csv')
#data = custom_dataset('./AsoSoftminiLabel.txt','./output3.csv')

train_dataset = []
for vector, label, length, context in data:
    train_dataset.append((label, vector, length,context))



#*****************************parameters ******************************

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EMBED_DIM = 32
NUN_CLASS = 6
VOCAB_SIZE = len(data.vocabulary)
N_EPOCHS = 5
min_valid_loss = float('inf')


#*********************************model ********************************

class TextClassification(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

model = TextClassification(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)

#**************************optimizer, loss function ********************

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)


#****************************split to train, valid***************************
train_len = int(len(train_dataset) * 0.95)

sub_train_, sub_valid_ = random_split\
    (train_dataset, [train_len, len(train_dataset) - train_len])

#***************************functions **************************************

def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


def train_func(sub_train_):
    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    for text, offsets, cls in data:
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)

        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return (train_loss / len(sub_train_)), (train_acc / len(sub_train_))


def test(validTest):
    loss = 0
    acc = 0
    data = DataLoader(validTest, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(validTest), acc / len(validTest)


#*****************************main**********************************
for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    print("run train_func")
    valid_loss, valid_acc = test(sub_valid_)
    print("run test")

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

#**************************test with random sample in validation*******************************
labels=data.labels
inverselabels = dict(map(reversed, labels.items()))

def test_with_validation():
    nums = range(0, len(sub_valid_))
    randoms = random.sample(nums,5)

    for i in randoms:
        label , vector, length, context = sub_valid_[i]
        target = inverselabels[str(label)]
        # BATCH_SIZE =1
        # data = DataLoader(sub_train_[i], batch_size=BATCH_SIZE, shuffle=True,
        #                   collate_fn=generate_batch)

        # text, offsets, cls = data
        off = torch.tensor([0])
        vector, off= vector.to(device), off.to(device)
        with torch.no_grad():
            predict = model(vector,off)
            predict=predict.argmax(1).item()
            predict= inverselabels[str(predict)]
        print('**********','output','*****************')
        print(context,'\n','target :',target,'\n','predict :',predict,'\n')

test_with_validation()
















#**********************************************************************************************************************
# print('text :',text,'\n','offsets:',offsets,'\n','cls:',cls)
#print('*************************************')
# for i in randoms:
#     textIndex = text[i]
#     textLabel = cls[i]
#     textoffsets = offsets[i]
#     with torch.no_grad():
#         output = model(text, offsets)
#     print("sample:", '\n', textIndex, '\n', 'target :', inverselabels[textLabel], '\n', 'predict:', '\n',
#           inverselabels[output])
#     print('*********************************************')


# # ListOfRandomValidation.append(sample)
# data = DataLoader(sample, batch_size=BATCH_SIZE, collate_fn=generate_batch)
# for text, offsets, cls in data:
#     text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
#     with torch.no_grad():
#         target = cls
#         output = model(text, offsets)
#     print("sample:",'\n',text,'\n','target :',inverselabels[target],'\n','predict:','\n',inverselabels[output])



# def test2():
#     nums = range(0, len(sub_valid_))
#     randoms = random.sample(nums,3)
#     print('len(sub_valid_):',len(sub_valid_))   #266
#     data = DataLoader(sub_valid_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
#     print('len(data):',len(data))       #17
#     print('next(iter(data)):',next(iter(data)))
#     textt =[]
#     offsetss = []
#     clss =[]
#     print('*************************************************')
#     for i, (text, offsets, cls) in enumerate(data):
#         text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
#         i
#         # print(i)
#         # print(text)
#         # print(len(text))
#         # print(text[1])
#         textt.append(text)
#         offsetss.append(offsets)
#         clss.append(cls)
#         print('*********************************************')
#     print(offsetss[1])
#     print(clss[1])
#     print(textt[1])



# #*********************************************************************
# # print(sub_valid_)
# # print(sub_valid_[1])
# # print(len(sub_valid_))
# #test2()
#
# #print(test2())
#
# nums = range(0, len(sub_valid_))
# randoms = random.sample(nums, 3)
# for i in randoms:
#     print('******************')
#     print(sub_valid_[i])



#**************************regulared data*******************************

# def regularedData(data):
#     train_dataset = []
#     for vector, label, length in data:
#         vector = np.array(vector)
#         vector = np.delete(vector, np.where(vector == 'unk')[0], axis=0)
#         length = len(vector)
#         vector = vector.astype(np.int)
#         vector = torch.LongTensor(vector)
#         # print(label)
#         label = int(label)
#         train_dataset.append((label, vector,length))
#
#     return train_dataset
#
# train_dataset = regularedData(data)


 # sampleIndexVector = np.array(sampleIndexVector)
        # sampleIndexVector = np.delete(sampleIndexVector, np.where(sampleIndexVector == 'unk')[0], axis=0)
        # samplelengthSample = len(sampleIndexVector)
        # sampleIndexVector = sampleIndexVector.astype(np.int)
        # sampleIndexVector = torch.LongTensor(sampleIndexVector)
        # # print(label)
        # sampleIndexlabel = int(sampleIndexlabel)










