"""
Copyright © 2023 Korea Testing Laboratory

Codes written by Yongyoon Lee
"""
##################################################
# 출처: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
import os
import csv
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

##########################
# 인공지능 알고리즘 클래스 #
class CMyAIAlgorithm:   

    def __init__(self, batchSize = 1, epochs = 2, dataPath = "./data/",\
                 trainDataFileName = "ktl_training_set.csv",\
                 testDataFileName = "ktl_testing_set.csv") -> None:
        self.batchSize = batchSize
        self.dataPath = dataPath
        self.epochs = epochs
        self.classes = []
        self.transform = []
        self.trainset = []
        self.testset = []
        self.trainloader = []
        self.testloader = []

        self.testDataPath = self.dataPath + "test_data/"      # 테스트 데이터 경로 지정
        os.makedirs(self.testDataPath, exist_ok=True)    # 테스트 데이터 폴더 생성, 이미 존재하면 오류를 발생하지 않음
        self.trainDataPath = self.dataPath + "train_data/"    # 학습 데이터 경로 지정
        os.makedirs(self.trainDataPath, exist_ok=True)   # 학습 데이터 폴더 생성, 이미 존재하면 오류를 발생하지 않음
        self.algorithmDataPath = self.dataPath + "cifar_net.pth"    # 알고리즘을 저장할 경로 지정

        # KTL 학습/테스트 데이터 목록을 넘겨 주기 위한 데이터 파일명
        self.trainDataFileName = trainDataFileName
        self.testDataFileName = testDataFileName

        # CNN 생성
        self.net = CNet()   # Convolutional Neural Network

        # Loss function 및 optimizer를 정의하기
        self.criterion = nn.CrossEntropyLoss()  # loss function
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)   # optimizer
   
    def getData(self):
        torchvision.datasets.CIFAR10.url="http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            
        self.trainset = torchvision.datasets.CIFAR10(root=self.dataPath, train=True,
                            download=True, transform=self.transform)
        self.testset = torchvision.datasets.CIFAR10(root=self.dataPath, train=False,
                            download=True, transform=self.transform)

        self.trainloader = torch.utils.data.DataLoader(dataset=self.trainset, batch_size=self.batchSize,
                            shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(dataset=self.testset, batch_size=self.batchSize,
                            shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 숫자로 된 레이블을, 해당 열 위치에 "1"로 표현함(문자 '0' 또는 '1')
    def __whichClasses__(self, idx: int)->list:
        out = ['0']*len(self.classes)
        out[idx] = '1'
        return out
    
    # 숫자로 된 레이블을, 해당 열 위치에 1로 표현함(숫자 0 또는 1)
    def __NwhichClasses__(self, idx: int)->list:
        out = [0]*len(self.classes)
        out[idx] = 1
        return out

    def saveDataToTestImages(self):
        testDataList = []
        testDataList.append(["File Path","File Name"] + list(self.classes))

        try:
            for idx, data in enumerate(tqdm(self.testloader, desc="- Saving test data")):
                image, label = data
                image = image * 0.5 + 0.5 # unnormalize
                save_image(image, self.testDataPath + f"test_image{idx:0>5}.png")

                # 테스트 데이터 목록을 만들기
                # path, fileName, [labels]...
                testDataList.append([self.testDataPath, f"test_image{idx:0>5}.png"]\
                                    + self.__whichClasses__(label.tolist()[0]))

        except KeyboardInterrupt:
            print("Ctrl + C 중지: saveDataToTestImages")

        # 테스트 데이터 목록을 파일로 저장하기
        # path, fileName, [labels]...
        with open(self.testDataFileName, 'w', newline='') as f:
            wr = csv.writer(f)

            for line in testDataList:
                wr.writerow(line)

    def saveDataToTrainImages(self):
        trainDataList = []
        trainDataList.append(["File Path","File Name"] + list(self.classes))

        try:
            for idx, data in enumerate(tqdm(self.trainloader, desc="- Saving train data")):
                image, label = data
                image = image * 0.5 + 0.5 # unnormalize
                save_image(image, self.trainDataPath + f"train_image{idx:0>5}.png")

                # 학습 데이터 목록을 만들기
                # path, fileName, [labels]...
                trainDataList.append([self.trainDataPath, f"train_image{idx:0>5}.png"]\
                                    + self.__whichClasses__(label.tolist()[0]))
                
        except KeyboardInterrupt:
            print("Ctrl + C 중지: saveDataToTrainImages")

        # 학습 데이터 목록을 파일로 저장하기
        # path, fileName, [labels]...
        with open(self.trainDataFileName, 'w', newline='') as f:
            wr = csv.writer(f)

            for line in trainDataList:
                wr.writerow(line)

    # 네트워크를 학습시키기
    def trainNet(self):
        print('===== Start Training =====')
        print("epoch, minibatch")
        try:
            for epoch in range(self.epochs):  # loop over the dataset multiple times
                running_loss = 0.0
                for i, data in enumerate(self.trainloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    if i % 2000 == 1999:    # print every 2000 mini-batches
                        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                        running_loss = 0.0
        except KeyboardInterrupt:
            print("Ctrl + C 중지: Training")
            exit()

        print('===== Finished Training =====')

    def saveNet(self):
        # 학습이 끝난 후, 모델을 저장함
        torch.save(self.net.state_dict(), self.algorithmDataPath)
    
    def loadNet(self):
        # 저장한 모델을 로드함
        try:
            self.net.load_state_dict(torch.load(self.algorithmDataPath))
        except:
            print("Cannot load algorithm file to Neural Network!")
            exit()

    def predictFromNet(self):
        # 모델로부터 예측값을 추론함
        groundTruth = []
        predictedProbability = []
        
        print('=== AI is inferring... ===')
        for data in tqdm(self.testloader, desc="- Predict test data"):
            test_image, GT_label = data
            predicted = self.net(test_image)

            groundTruth.append(self.__NwhichClasses__(GT_label.tolist()[0]))
            predictedProbability.append(predicted.tolist()[0])

        return groundTruth, predictedProbability

    def doClassification(self):
        # 인공지능에서 추론한 확률 값을 분류로 확정하여,
        # Positive(1)/Negative(0)로 판정함
        # 결과를 predictedProbability 리스트에 다시 저장함
        groundTruth = []
        predictedClassification = []
        
        print('=== AI is inferring(classification)... ===')
        for data in tqdm(self.testloader, desc="- Predict test data"):
            test_image, GT_label = data
            predicted = self.net(test_image)

            groundTruth.append(self.__NwhichClasses__(GT_label.tolist()[0]))
            # 추론한 확률이 가장 높은 값을 분류로 선택함
            a_list = predicted.tolist()[0]
            predictedClassification.append(self.__NwhichClasses__(a_list.index(max(a_list))))

        return groundTruth, predictedClassification

##################################################
# Convolutional Neural Network를 정의하기
#-------------------------------------------------
# kernal_size =5의 convolution 층 2개, max_pooling 층 2개,
# fully_connected 층 3개로 이루어짐.
# 마지막 층의 활성함수가 softmax가 아니라 linear임
class CNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x