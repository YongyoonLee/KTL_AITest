"""
Copyright © 2023 Korea Testing Laboratory

Codes written by Yongyoon Lee
"""
from .KTL_DataOverlaps import DataOverlaps
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import csv
from tqdm import tqdm

# Class AiTest
# 인공지능 알고리즘을 테스트하기 위한 상위 클래스
# Ai 테스트를 위해, 본 상위 클래스를 상속 받아, 인공지능 성능 테스트를 수행한다.
class AITest():

    def __init__(self, trainingFileName: str, testingFileName: str, diseaseList: list, testResultsFileName="test_results.csv"):
        self.trainingFileName = trainingFileName
        self.testingFileName = testingFileName
        self.hashedTrainingFileName = ""
        self.hashedTestingFileName = ""
        self.diseaseList = diseaseList  # diseaseList["disease1", "disease2", ...]

        self.hasOverlaps = False

        self.predictedProbability = [] # predictedProbability[disease1_prob, disease2_prob, ...]
        self.predictedClassification = [] # predictedClassification[disease1_class, disease2_class, ...]
        self.groundTruth = []          # groundTruth[disease1_true, disease2_true, ...]
        self.aucList = []              # auc value
        self.TPTNFPFN = []             # classification results["TP","TN","FP","TP", ...]
        self.confusionMatrix = []      # confusion matrix[[TP, TN, FP, FN],[TP, TN, FP, FN],...]

        self.testResultsFileName = testResultsFileName  # 정답값과 인공지능 추론 결과를 저장하기 위한 파일 이름

    # 1. Hash 값을 계산하여, hashed_파일로 저장한다.
    # 2. 중복 값이 있는지 확인하여, hasOverlaps에 True/False를 저장한다.
    def doHashAndSave(self):
        
        dataOverlaps = DataOverlaps(self.trainingFileName, self.testingFileName)

        self.hashedTrainingFileName, self.hashedTestingFileName = dataOverlaps.calculateHashAndSave() # Hash 값을 계산하여, hashed_파일로 저장하고, 파일 이름을 반환한다.

        self.hasOverlaps = dataOverlaps.hasOverlaps() # 중복 값이 있는지 확인한다.

    # 이 메서드를 상속받아 구현하여 사용해야 함
    def doAITest(self):
        # test 데이터 목록을 읽어 와, 알고리즘에 입력하여, 테스트를 수행한다.
        # 입력: self.testingFileName: test 데이터 목록
        # 출력: self.predictedProbability: 2차원 리스트
        pass

    def drawROC(self):
        array_groundTruth = np.array(self.groundTruth)  # 정답값을 numpy array로 변환
        array_predictedProbability = np.array(self.predictedProbability)    # 인공지능 추론 결과를 numpy array로 변환

        for i, disease in enumerate(tqdm(self.diseaseList, desc='-- drawROC')):       
            ith_groundTruth = array_groundTruth[:, i]   # i번째 열만 추출함
            ith_predictedProbability = array_predictedProbability[:, i] # i번째 열만 추출함

            false_positive_rate, true_positive_rate, threshholds = roc_curve(ith_groundTruth, ith_predictedProbability)

            #roc_auc = auc(false_positive_rate, true_positive_rate)

            plt.title('Receiver Operating Characteristic for '+str(disease))
            plt.xlabel('False Positive Rate(1 - Specificity)')
            plt.ylabel('True Positive Rate(Sensitivity)')

            plt.plot(false_positive_rate, true_positive_rate, 'b')
 
            plt.plot([0,1], [0,1], 'r--')

            plt.savefig("ROC_" + str(disease) + ".png", dpi = 1000)
            plt.close()

    def calculateAUC(self, aucResultFileName: str = "auc_result.txt"):
        with open(aucResultFileName, 'w') as file: # auc 값을 계산하고, 각 질병 별 auc 값과 평균 auc 값을 텍스트 파일에 저장함
            array_groundTruth = np.array(self.groundTruth)  # 정답값을 numpy array로 변환
            array_predictedProbability = np.array(self.predictedProbability)    # 인공지능 추론 결과를 numpy array로 변환

            for i, disease in enumerate(tqdm(self.diseaseList, desc='-- calculateAUC')):
                ith_groundTruth = array_groundTruth[:, i]   # i번째 열만 추출함
                ith_predictedProbability = array_predictedProbability[:, i] # i번째 열만 추출함

                false_positive_rate, true_positive_rate, threshholds = roc_curve(ith_groundTruth, ith_predictedProbability)

                roc_auc = auc(false_positive_rate, true_positive_rate)
                self.aucList.append(roc_auc)
            
                file.writelines("The auc value of "+str(disease)+": "+str(roc_auc)+"\n")
            
            file.writelines("==================================================\n")
            file.writelines("The average auc value: "+ str(np.mean(self.aucList)))

    def calculateConfusionMatrix(self): # classification을 계산하여, TP, TN, FP, FN의 결과를 리스트로 반환함
        trueFalse = []
        positiveNegative = []
        TPTNFPFN = []
        
        dimension = len(np.array(self.groundTruth).shape)

        # 1. TP, TN, FP, FN을 판정함
        #    TP: Ground Truth와 Predicted Class가 같고, Predicted Class가 Positive인 경우
        #    TN: Ground Truth와 Predicted Class가 같고, Predicted Class가 Negative인 경우
        #    TP: Ground Truth와 Predicted Class가 같고, Predicted Class가 Positive인 경우
        #    FP: Ground Truth와 Predicted Class가 다르고, Predicted Class가 Positive인 경우
        #    FN: Ground Truth와 Predicted Class가 다르고, Predicted Class가 Negative인 경우
        if (dimension == 1):
        # predictedClassification 리스트가 1차원일 경우, binary class를 계산하여, TP, TN, FP, FN의 결과를 리스트로 반환함
            for idx, groundTruth in enumerate(self.groundTruth):
                trueFalse.append("T") if groundTruth == self.predictedClassification[idx] else trueFalse.append("F")
                positiveNegative.append("P") if self.predictedClassification[idx] == 1 else positiveNegative.append("N")
                TPTNFPFN.append(trueFalse[idx]+positiveNegative[idx])
        else:
        # predictedClassification 리스트가 2차원 이상일 경우, 각 열 별로 TP, TN, FP, FN의 결과를 리스트로 반환함
            for idx, groundTruths in enumerate(self.groundTruth):
                trueFalse.clear()
                positiveNegative.clear()
                tFpN = []   # 리스트 객체 추가는 얕은 복사를 하기 때문에, 리스트 객체를 새로 생성함
                for jdx, groundTruth in enumerate(groundTruths):
                    trueFalse.append("T") if groundTruth == self.predictedClassification[idx][jdx] else trueFalse.append("F")
                    positiveNegative.append("P") if self.predictedClassification[idx][jdx] == 1 else positiveNegative.append("N")
                    tFpN.append(trueFalse[jdx]+positiveNegative[jdx])
                TPTNFPFN.append(tFpN)   # 주의! 깊은 복사를 사용하거나, 리스트 객체를 새로 생성해서 할당해야 함

        self.TPTNFPFN = TPTNFPFN
        self.__saveTestClassResults__()      # 정답값과 인공지능 추론 결과(classification)를 파일에 저장함

        # 2. TP, TN, FP, FN의 개수를 세어서 [[TP수, TN수, FP수, FN수], [TP수, TN수, FP수, FN수],...]으로 리턴함
        if (dimension == 1):
        # predictedClassification 리스트가 1차원일 경우
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            for item in TPTNFPFN:
                if item == 'TP':
                    TP += 1
                elif item == 'TN':
                    TN += 1
                elif item == 'FP':
                    FP += 1
                elif item == 'FN':
                    FN += 1
            self.confusionMatrix = [TP, TN, FP, FN]     # self.confusionMatrix에 혼동행렬 값을 저장함
        else:
            numClass = len(self.diseaseList)    # 질병 개수
            confusionMatrix = []
            for i in range(numClass):
                confusionMatrix.append([0, 0, 0, 0])  # [[TP수, TN수, FP수, FN수], [TP수, TN수, FP수, FN수],...]

            for idx, items in enumerate(TPTNFPFN):
                for jdx, item in enumerate(items):
                    if item == 'TP':
                        confusionMatrix[jdx][0] += 1
                    elif item == 'TN':
                        confusionMatrix[jdx][1] += 1
                    elif item == 'FP':
                        confusionMatrix[jdx][2] += 1
                    elif item == 'FN':
                        confusionMatrix[jdx][3] += 1

            self.confusionMatrix = confusionMatrix  # self.confusionMatrix에 혼동행렬 값을 저장함

        self.__saveConfusionMatrix__()       # confusionMatrix값을 파일로 저장함

    def __printTestResults__(self):
        # 정답값과 인공지능 추론결과를 화면에 출력하기
        print("=== Predicted Probability ===")
        for item in self.predictedProbability:
            print(item)

        print("=== Ground Truth ===")
        for item in self.groundTruth:
            print(item)

    def __saveTestResults__(self):
        # 정답값과 인공지능 추론결과를 csv 파일로 저장하기
        # [Predicted Probibilites 1, 2, 3, ...], [Ground Truths 1, 2, 3, ...]
        with open(self.testResultsFileName, 'w', newline='') as f:
            wr = csv.writer(f)

            # 첫행 저장
            line = []
            for item in self.diseaseList:
                line.append("GT " + item)
            for item in self.diseaseList:
                line.append("Predicted " + item)
            wr.writerow(line)

            # 정답값과 인공지능 추론 결과를 문자열로 변환하여 저장함
            for idx, predict in enumerate(self.predictedProbability):
                line = list(map(str, self.groundTruth[idx])) + list(map(str, predict))
                wr.writerow(line)
    
    def __saveTestClassResults__(self, testClassResultsFileName='test_results_class.csv'):
        # 정답값과 인공지능 추론결과를 csv 파일로 저장하기
        # [Predicted Probibilites 1, 2, 3, ...], [Ground Truths 1, 2, 3, ...], [TP,TN,FP,...]
        with open(testClassResultsFileName, 'w', newline='') as f:
            wr = csv.writer(f)

            # 첫행 저장
            line = []
            for item in self.diseaseList:
                line.append("GT " + item)
            for item in self.diseaseList:
                line.append("Predicted " + item)
            for item in self.diseaseList:
                line.append("Eval " + item)
            wr.writerow(line)

            # 정답값과 인공지능 추론 결과를 문자열로 변환하여 저장함
            for idx, predict in enumerate(self.predictedClassification):
                line = list(map(str, self.groundTruth[idx])) + list(map(str, predict)) + list(map(str, self.TPTNFPFN[idx]))
                wr.writerow(line)
    
    def __saveConfusionMatrix__(self, saveFileName='confusion_matrix.txt'):
        # 혼동행렬 결과값 저장
        with open(saveFileName, 'w', newline='') as f:
            for idx, disease in enumerate(self.diseaseList):
                f.write('=== ' + disease + ' ===\n')
                TP = self.confusionMatrix[idx][0]
                TN = self.confusionMatrix[idx][1]
                FP = self.confusionMatrix[idx][2]
                FN = self.confusionMatrix[idx][3]
                f.write('TP: '+str(TP)\
                        +', TN: '+str(TN)+"\n"\
                        +'FP: '+str(FP)\
                        +', FN: '+str(FN)+"\n")
                sensitivity = TP / (TP + FN)
                specificity = TN / (TN + FP)
                accuracy = (TP + TN) / (TP + FP + FN + TN)
                PPV = TP / (TP + FP)
                NPV = TN / (TN + FN)
                FScore = (2 * PPV * sensitivity) / (PPV + sensitivity)
                f.write('sensitivity(TPR): '+str(sensitivity)+"\n"\
                        'specificity: '+str(specificity)+"\n"\
                        'accuracy: '+str(accuracy)+"\n"\
                        'precision(PPV): '+str(PPV)+"\n"\
                        'NPV: '+str(NPV)+"\n"\
                        'FScore: '+str(FScore)+"\n")