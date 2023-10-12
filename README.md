# KTL_AITest

## 1. 설정
처음 사용 시, 다음 명령어로 필요 라이브러리를 설치한다.
```console
pip install -r requirements.txt
```
윈도우 사용자의 경우, ms 런타임이 필요할 수 있다.
```console
pip install msvs-runtime
```

## 2. 코드 목록
* KTL에서 제공하는 코드
  * KTL_AI/KTL_AITest.py
    * AITest 클래스 : 인공지능의 functional performance 테스트를 수행함
  * KTL_AI/KTL_DataOverlaps.py
    * DataOverlaps 클래스 : 학습 데이터와 테스트 데이터에 서로 중복된 데이터가 있는지 확인함
  * KTL_AI/KTL_FileListToHash.py
    * LoadFileListToHash 클래스 : 학습 데이터 목록과 테스트 데이터 목록을 읽어서, 타임스탬프, SHA256, MD5를 추가하여 저장함
  * KTL_AI/KTL_MeasureElapsedTime.py
    * ElapsedTime 클래스 : 테스트 경과 시간을 측정함
* 사용자 코드(여기서는 예시임, 클래스명을 바꿔서 사용할 수 있음. My_AITest -> KNU_AITest 등)
  * MyAIAlgorithm.py
    * CAIAlgorithm 클래스 : 사용자가 만든 인공지능 알고리즘
    * CNet 클래스 : 사용자가 사용하는 인공지능 네트워크
  * MyAITest.py
    * My_AITest 클래스 : 사용자가 만든 인공지능 알고리즘을 KTL에서 제공하는 AITest 클래스를 상속 받아 테스트를 수행함

## 3. 테스트 시 입력해야 하는 데이터
* 파일로 전달(My_AITest 생성자에서 파일명을 바꿀 수 있음)
  * ktl_training_set.csv : 인공지능 학습 데이터 목록
  * ktl_testing_set.csv : 인공지능 테스트 데이터 목록
* My_AITest 클래스의 부모 클래스인 AITest의 프로퍼티로 전달
  * ROAUC 값 계산 시
    * self.GroundTruth : 테스트 데이터의 정답값 목록, list 타입(2차원일 경우, 1차원에 테스트 데이터 하나에 대한 정답값들을 넣음)
    * self.predictedProbability : 인공지능이 예측한 확률값 목록, list 타입(2차원일 경우, 1차원에 테스트 데이터 하나에 대한 예측값들을 넣음)
  * Classification 값 계산 시
    * self.GroundTruth : 테스트 데이터의 정답값 목록, list 타입(2차원일 경우, 1차원에 테스트 데이터 하나에 대한 정답값들을 넣음)
    * self.predictClassification : 인공지능이 예측한 클래스 목록, list 타입(2차원일 경우, 1차원에 테스트 데이터 하나에 대한 클래스[0: 음성, 1: 양성]들을 넣음)

## 4. 테스트 실행 방법
1. MyAITest.py 파이썬 파일을 직접 실행
   ```console
   python MyAITest.py
   ```
2. pytest를 사용하여, 합격/불합격 판정을 수행하면서 실행
   ```console
   pytest -sv MyAITest.py
   ```