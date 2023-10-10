"""
Copyright © 2023 Korea Testing Laboratory

Codes written by Yongyoon Lee
"""
from time import time
from datetime import datetime
from datetime import timedelta

'''
class ElapsedTime 사용법


elapsedTime = ElapsedTime() # 시작 시각 설정

# 실행 시간을 알고자 하는 코드 삽입

elapsedTime.printElapsedTime() # 종료 시각을 알아와서, 실행 시간을 계산하여, 화면에 표시해 줌

'''
class ElapsedTime:
	def __init__(self):
		self.startTime = time()
		print("The test started at " + str(datetime.now()))
		
	def printElapsedTime(self):
		self.endTime = time()
		elapsedTime = self.endTime - self.startTime
		result = timedelta(seconds=elapsedTime)
		print("The test has run for " + str(result))


if __name__ == '__main__':
        elapsedTime = ElapsedTime()
        elapsedTime.printElapsedTime()