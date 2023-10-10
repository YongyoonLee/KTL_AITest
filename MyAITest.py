"""
Copyright © 2023 Korea Testing Laboratory

Codes written by Yongyoon Lee
"""
import os
from KTL_AI.KTL_AITest import AITest
from KTL_AI.KTL_MeasureElapsedTime import ElapsedTime
from MyAIAlgorithm import CMyAIAlgorithm

# KTL에서 제공한 AITest 클래스를 상속 받아와, 나의 AITest 클래스를 정의함
class My_AITest(AITest):

    def __init__(self, trainingFileName: str, testingFileName: str, diseaseList: list, testResultsFileName):
        super().__init__(trainingFileName, testingFileName, diseaseList, testResultsFileName)
        
        # 나의 인공지능 알고리즘 클래스 객체를 생성함
        self.My_AIAlgorithm = CMyAIAlgorithm()

        # 1. 학습/테스트 데이터를 다운로드하고, 학습/테스트를 위한 데이터 셋을 준비함
        self.My_AIAlgorithm.getData()    # CIFAR10를 (다운)로드하고 정규화하기
        self.My_AIAlgorithm.saveDataToTestImages()   # 테스트 데이터를 png 파일로 저장하기 & 테스트 데이터 목록도 저장하기
        self.My_AIAlgorithm.saveDataToTrainImages()  # 학습 데이터를 png 파일로 저장하기 & 학습 데이터 목록도 저장하기

    def doAITest(self, mode="for AUC"):
        # test 데이터 목록을 읽어 와, 알고리즘에 입력하여, 테스트를 수행한다.
        # 입력: self.testingFileName: 테스트 데이터 목록 <- 이 예제에서는 이미 My_AIAlgorithm 객체에서
        #                                                  테스트 데이터 셋을 불러 왔으므로
        #                                                  굳이 테스트 데이터 목록을 또 읽어 올 필요가 없이,
        #                                                  이미 불러온 테스트 데이터 셋을 사용하여 테스트를 수행함
        # 출력: self.groundTruth: 2차원 리스트
        #       self.predictedProbability: 2차원 리스트 (확률로 출력)
        #       self.predictedClassification: 2차원 리스트 (분류로 출력) - 1: Positive, 0: Negative
        
        # 2. 나의 인공지능 알고리즘 객체에서, 인공지능 학습을 수행함
        if not os.path.isfile(self.My_AIAlgorithm.algorithmDataPath):  # 아직 학습을 수행하지 않아, 알고리즘 파일이 존재하지 않는 경우
            self.My_AIAlgorithm.trainNet()      # Neural Network를 생성하고, 학습 데이터로 Neural Network를 학습시킴
            self.My_AIAlgorithm.saveNet()       # 학습을 완료한 Neural Network 알고리즘을 저장함

        # 3. 이미 학습한 인공지능 알고리즘을 로드함
        self.My_AIAlgorithm.loadNet()    # 저장한 Neural Network 알고리즘을 로드함

        # 4-1. 테스트 데이터 셋을 입력하여, 인공지능 추론 결과(확률)를 뽑음
        if mode == "for AUC":
            self.groundTruth, self.predictedProbability = self.My_AIAlgorithm.predictFromNet() # 테스트 데이터를 Neural Network에 입력하여, 인공지능이 예측한 값(Probability)을 가져옴
            super().__saveTestResults__()   # 정답값과 인공지능 추론 결과를 파일에 저장함
        # 4-2. Classification(확정된 값)을 수행하는 경우
        elif mode == "for classification":
            self.groundTruth, self.predictedClassification = self.My_AIAlgorithm.doClassification() # 테스트 데이터를 Neural Network에 입력하여, 인공지능이 예측한 값(Classification)을 가져옴

if __name__ == '__main__':

    elapsedTime = ElapsedTime()
        
    MyAITest = My_AITest('ktl_training_set.csv', 'ktl_testing_set.csv',\
                           ['plane', 'car', 'bird', 'cat',\
                            'deer', 'dog', 'frog', 'horse',\
                             'ship', 'truck'], "test_results.csv")

    MyAITest.doHashAndSave()
    
    print("Overlaped Data Flag: " + str(MyAITest.hasOverlaps)) # 중복 데이터 확인
    
    # 1. AI가 추론한 결과값이 Probability로 나오는 경우 -> AUC 값 계산
    MyAITest.doAITest("for AUC") # 실제 Test run
    
    MyAITest.drawROC() # ROC 곡선 그림
    
    MyAITest.calculateAUC() # AUC 값 계산

    # 2. AI가 추론한 결과값이 Classification으로 나오는 경우 -> confusion matrix 값 계산
    MyAITest.doAITest("for classification") # 실제 Test run

    MyAITest.calculateConfusionMatrix()  # Confusion Matrix를 계산함, Classification 결과와 ConfusionMatrix를 저장함
    
    elapsedTime.printElapsedTime()

def test001_averageAUC():
    MyAITest = My_AITest('ktl_training_set.csv', 'ktl_testing_set.csv',\
                           ['plane', 'car', 'bird', 'cat',\
                            'deer', 'dog', 'frog', 'horse',\
                             'ship', 'truck'], "test_results.csv")

    MyAITest.doHashAndSave()
    
    print("Overlaped Data Flag: " + str(MyAITest.hasOverlaps)) # 중복 데이터 확인
    
    MyAITest.doAITest("for AUC") # 실제 Test run
    
    MyAITest.drawROC() # ROC 곡선 그림
    
    MyAITest.calculateAUC() # AUC 값 계산
    
    assert (sum(MyAITest.aucList)/len(MyAITest.aucList)) >= 0.8     # 평균 auc가 0.8 이상인지 확인