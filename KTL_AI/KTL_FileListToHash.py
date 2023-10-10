"""
Copyright © 2023 Korea Testing Laboratory

Codes written by Yongyoon Lee
"""
import hashlib
from datetime import datetime
from tqdm import tqdm

# Class LoadFileListToHash
# 파일 목록을 읽어 와, time stamp, SHA256, MD5 해쉬 코드를 추가함.
# input: fileList: list = [[file path], [file name], [disease1], [disease2], ...]
# output: fileList: list = [[time stamp], [file path], [file name], [disease1], [disease2], ..., [SHA256], [MD5]]
class LoadFileListToHash():
    
    def __init__(self, fileList: list):
        self.__fileList = fileList

    # Method __fileToMD5__
    # 파일을 읽어 MD5 해쉬 코드를 생성하는 메소드
    def __fileToMD5__(self, filename: str, blocksize: int = 4096) -> str:
        hsh = hashlib.md5() # MD5 객체를 생성함

        try:
            with open(filename, "rb") as f: # 파일을 바이너리로 읽음
                while True:
                    buf = f.read(blocksize) # 블럭 단위로 파일을 읽어 옴
                    if not buf:             # 파일을 다 읽어 오면, 루프를 종료함
                        break
                    hsh.update(buf)         # 버퍼 내용으로 해쉬 값을 생성함

        except:
            print("Cannot open file.")
            exit()
        
        return hsh.hexdigest()          # 해쉬 코드를 16진수로 반환함

    # Method __fileToSHA256__
    # 파일을 읽어 SHA256 해쉬 코드를 생성하는 메소드
    def __fileToSHA256__(self, filename: str, blocksize: int = 4096) -> str:
        hsh = hashlib.sha256() # SHA256 객체를 생성함

        try:
            with open(filename, "rb") as f: # 파일을 바이너리로 읽음
                while True:
                    buf = f.read(blocksize) # 블럭 단위로 파일을 읽어 옴
                    if not buf:             # 파일을 다 읽어 오면, 루프를 종료함
                        break
                    hsh.update(buf)         # 버퍼 내용으로 해쉬 값을 생성함

        except:
            print("Cannot open file.")
            exit()
        
        return hsh.hexdigest()          # 해쉬 코드를 16진수로 반환함

    # Method addTimeStampSHA256MD5
    # 프로퍼티 self.__fileList를 읽어 와, 한 행씩, TimeStamp를 insert, SHA256를 append, MD5를 append,
    # 그 결과를 반환하는 메소드
    def addTimeStampSHA256MD5(self, description: str) -> list:

        for i, fileList in enumerate(tqdm(self.__fileList, desc=("--"+description+": addTimeStampSHA256MD5()"))):
            fileName = fileList[0] + '/' + fileList[1]

            # 1. TimeStamp를 추가함
            fileList.insert(0, str(datetime.now()))

            # 2. SHA256을 추가함
            fileList.append(self.__fileToSHA256__(filename = fileName))

            # 3. MD5를 추가함
            fileList.append(self.__fileToMD5__(filename = fileName))

        return self.__fileList

if __name__ == '__main__':
    # Class LoadFileListToHash 예제
    import csv
    # 1. training_files.csv 파일을 읽어 와, loadedTrainingFiles 리스트에 저장한다.
    loadedTrainingFileList = []
    addedTrainingFileList = []
    with open('training_files.csv', 'r') as f:
        rdr = csv.reader(f)

        for line in rdr:
            loadedTrainingFileList.append(line)

    # 2. LoadFileListToHash 클래스 인스턴스를 생성하여, loadedTrainingFileList 리스트(2번째 행부터)를 입력한다.
    loadFileListToHash = LoadFileListToHash(loadedTrainingFileList[1:])
    
    # 3. loadFileListToHash 인스턴스에서 addTimeStampSHA256MD5 메소드를 실행한다.
    addedTrainingFileList = loadFileListToHash.addTimeStampSHA256MD5()

    # 4. 첫 행에, 각 열에 대한 속성 값을 넣는다.
    addedTrainingFileList.insert(0, ["Time Stamp", "File Path", "File Name", "SHA256", "MD5"])
    
    with open('added_training_files.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        for line in addedTrainingFileList:
            wr.writerow(line)