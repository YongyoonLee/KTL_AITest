"""
Copyright © 2023 Korea Testing Laboratory

Codes written by Yongyoon Lee
"""
from .KTL_FileListToHash import LoadFileListToHash
import csv

# Class DataOverlaps
# 학습 파일에 대한 목록이 기록된 파일과, 테스팅 파일에 대한 목록이 기록된 파일을 읽어 와,
# SHA256과 MD5 해쉬 코드를 생성하고, 그 내용을 Result 파일에 저장하고,
# 학습 파일의 SHA256 목록과, 테스팅 파일의 SHA256 목록이 중복이 있는지 판단하여, 중복이 있을 경우 True를 반환함
# 학습 파일의 MD5 목록과, 테스팅 파일의 MD5 목록이 중복이 있는지 판단하여, 중복이 있을 경우 True를 반환함
class DataOverlaps():

    def __init__(self, trainingFileName: str, testingFileName: str):

        self.__trainingData = []
        self.__testingData = []

        # 1. 학습 파일에 대한 목록이 기록된 파일과, 테스팅 파일에 대한 목록이 기록된 파일 이름을 가져와
        #    해쉬 코드 결과를 포함하여 기록할 파일 이름을 붙여 줌
        self.__trainingFileName = trainingFileName
        self.__testingFileName = testingFileName

        splittedFileName = trainingFileName.split('/')
        splittedFileName[-1] = 'hashed_' + splittedFileName[-1]

        self.__hashedTrainingFileName = '/'.join(splittedFileName)

        splittedFileName = testingFileName.split('/')
        splittedFileName[-1] = 'hashed_' + splittedFileName[-1]

        self.__hashedTestingFileName = '/'.join(splittedFileName)

    # 메소드 __hasOverlaps
    # 두 개의 리스트를 입력 받아, 중복 되는 부분이 있으면 True를, 중복 되는 부분이 없으면 False를 반환하는 함수
    def __hasOverlaps(self, list1: list, list2: list) -> bool:
        addedList = list1 + list2 # 입력 받은 두 개의 리스트를 합쳐서 addedList에 저장한다.

        return len(addedList) != len(set(addedList)) # 합친 addedList에 중복 되는 값이 있으면 True를 반환하고, 중복되는 값이 없으면 False를 반환한다.

    # 메소드 calculateHashAndSave
    # 학습 파일에 대한 목록이 기록된 파일을 읽어 오고, 테스트 파일에 대한 목록이 기록된 파일을 읽어 와,
    # TimeStamp, SHA256, MD5를 계산하고, 그 결과를 포함하여 파일로 저장한다.
    def calculateHashAndSave(self):

        try:
            # 1. 학습 파일에 대한 목록이 기록된 파일을 읽어 와, __traniningData에 저장한다.
            with open(self.__trainingFileName, 'r') as f:
                rdr = csv.reader(f)

                for line in rdr:
                    self.__trainingData.append(line)

            # 2. 테스트 파일에 대한 목록이 기록된 파일을 읽어 와, __testingData에 저장한다.
            with open(self.__testingFileName, 'r') as f:
                rdr = csv.reader(f)

                for line in rdr:
                    self.__testingData.append(line)
        except:
            print("학습/테스트 파일 읽기와 관련하여 오류가 발생하였습니다.")
            exit()

        # 3. LoadFileListToHash 클래스 인스턴스를 생성하여, trainingData 리스트(2번째 행부터)를 입력한다.
        training_loadFileListToHash = LoadFileListToHash(self.__trainingData[1:])
        trainingDataHeader = self.__trainingData[0]

        # 4. LoadFileListToHash 클래스 인스턴스를 생성하여, testingData 리스트(2번째 행부터)를 입력한다.
        testing_loadFileListToHash = LoadFileListToHash(self.__testingData[1:])
        testingDataHeader = self.__testingData[0]

        # 5. 각각 addTimeStampSHA256MD5 메소드를 실행한다.
        self.__trainingData = training_loadFileListToHash.addTimeStampSHA256MD5("Training Data")
        self.__testingData = testing_loadFileListToHash.addTimeStampSHA256MD5("Testing Data")
        
        trainingDataHeader = ["Time Stamp"] + trainingDataHeader + ["SHA256", "MD5"]
        testingDataHeader  = ["Time Stamp"] + testingDataHeader  + ["SHA256", "MD5"]

        # 6. 첫 행에, 각 열에 대한 속성 값을 넣는다.
        self.__trainingData.insert(0, trainingDataHeader)
        self.__testingData.insert(0, testingDataHeader)
        
        # 7. 해쉬 코드 결과를 포함하여, 파일로 저장한다.
        with open(self.__hashedTrainingFileName, 'w', newline='') as f:
            wr = csv.writer(f)

            for line in self.__trainingData:
                wr.writerow(line)

        with open(self.__hashedTestingFileName, 'w', newline='') as f:
            wr = csv.writer(f)

            for line in self.__testingData:
                wr.writerow(line)
                
        # 8. 해쉬 파일 이름을 반환한다.
        return self.__hashedTrainingFileName, self.__hashedTestingFileName

    def hasOverlaps(self) -> bool:

        isOverlapedSHA256 = False
        isOverlapedMD5 = False

        listSHA256 = []
        listMD5 = []
        
        # 1. trainingData의 SHA256 값들과, testingData의 SHA256 값들 중에, 중복이 있는지 확인한다.
        i = self.__trainingData[0].index("SHA256")

        for line in self.__trainingData[1:]:
            listSHA256.append(line[i])

        j = self.__testingData[0].index("SHA256")
        for line in self.__testingData[1:]:
            listSHA256.append(line[j])

        isOverlapedSHA256 = (len(listSHA256) != len(set(listSHA256)))

        # 2. trainingData의 MD5 값들과, testingData의 MD5 값들 중에, 중복이 있는지 확인한다.
        i = self.__trainingData[0].index("MD5")

        for line in self.__trainingData[1:]:
            listMD5.append(line[i])

        j = self.__testingData[0].index("MD5")
        for line in self.__testingData[1:]:
            listMD5.append(line[j])

        isOverlapedMD5 = (len(listMD5) != len(set(listMD5)))

        # 3. 중복 결과를 반환한다.
        return isOverlapedSHA256 and isOverlapedMD5
        

if __name__ == '__main__':

    dataOverlaps = DataOverlaps('d:/training_files.csv', 'd:/testing_files.csv')

    dataOverlaps.calculateHashAndSave() # Hash 값을 계산하여, hashed_파일로 저장한다.


    print(dataOverlaps.hasOverlaps()) # 중복 값이 있는지 확인한다.