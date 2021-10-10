# SageMaker-Tensorflow-Step-by-Step 워크샵


## 1. 배경

TensorFlow™를 통해 개발자는 클라우드에서 딥러닝을 쉽고 빠르게 시작할 수 있습니다.
이 프레임워크는 다양한 산업 분아에서 사용되고 있으며 특히 컴퓨터 비전, 자연어 이해 및 음성 번역과 같은 영역에서 딥러닝 연구 및 응용 프로그램 개발에 널리 사용됩니다.
머신 러닝 모델을 대규모로 구축, 학습 및 배포 할 수있는 플랫폼인 Amazon SageMaker를 통해 완전히 관리되는(fully-managed) TensorFlow 환경에서 AWS를 시작할 수 있습니다.

## 2. 워크샵 목적

이 페이지의 하단의 참고 사항을 보면 좋은 예시 리소스가 많이 있습니다. 아래의 예시에서 빠진 내용 및 내용을 최신 업데이트하여 워크샵을 만들었습니다. 이 워크샵을 통해서 아래와 같은 점을 배우게 됩니다.

## 3. 배우는 내용 들

### [알림 사항]
- **<font color="red">[중요] 세이지 메이커 노트북 인스턴스에서 실행을 해야 합니다.</font>**
    - `노트북에서 "로컬모드" 를 제외하면 "세이지 메이커 스튜디오" 에서도 실행 가능합니다.`
    
    
- 이 워크샵은 Tensorflow 2.4.1 에서 동작이 되고, 이 버전에서 테스트 되었습니다. 

### [사용한 기술 요소들]
- 데이터 세트 준비
    - Cifar10 의 데이터 세트를 인터넷에서 원본 다운로드 후, TF Record 를 생성하는 과정


- Keras 버전
    - 세이지 메어커 사용 없이 Keras 버전의 훈련 코드 작성
    - 세이지 메어커 사용하는 Keras 버전의 훈련 코드 작성 (세이지 메이커 스크립트 코드)
    - 세이지 메이커의 로컬 모드 및 호스트 모드로 훈련
    - 세이지 메이커 Horovod (분산 훈련) 로 로컬 모드 및 호스트 모드로 훈련  
        - 호스트 모드시 2개의 EC2 인스턴스로 훈련


- TF2 버전
    - 세이지 메어커 사용 하는 TF2 버전의 훈련 코드 작성 (세이지 메이커 스크립트 코드)
    - 세이지 메이커의 로컬 모드 및 호스트 모드로 훈련
    - 세이지 메이커 Horovod (분산 훈련) 로 로컬 모드 및 호스트 모드로 훈련  
        - 호스트 모드시 2개의 EC2 인스턴스로 훈련
    - **<font color="red">[중요] 아래는 ml.p3.16xlarge 로 노트북 인스턴스가 필요합니다.</font>**
       - 세이지 메이커 분산 훈련 라이브러리 (DDP 사용) 로 로컬 모드 및 호스트 모드로 훈련  
           - 호스트 모드시 2개의 EC2 ml.p3.16xlarge 인스턴스로 훈련


## 4. 노트북 구성

- 메인 폴더: SageMaker-Tensorflow-Step-By-Step/code/phase0


- 0.0.Setup-Environment.ipynb 
    - conda_tensorflow2_p36 커널에 tensorflow==2.4.1 설치 및 기타 필요 패키지 설치


- 1.0.Downlaod-Dataset.ipynb
    - Cifar10 데이터 다운로드 및 TF Record 생성


- 1.1.Train-Scratch.ipynb
    - 스크래치 버전의 Keras 훈련 코드 노트북


- 1.2.Train_Keras_Local_Script_Mode.ipynb    
    - 세이지 메이커 Keras 훈련 코드 로컬 모드 및 호스트 모드 실행


- 1.5.Train_Keras_Horovod.ipynb    
    - 세이지 메이커 Keras 훈련 코드 호로보드 분산 훈련


- 2.1.Train_TF_Local_Script_Mode.ipynb    
    - 세이지 메이커 TF2 훈련 코드 로컬 모드 및 호스트 모드 실행


- 2.2.Train_TF_Horovod.ipynb
    - 세이지 메이커 TF2 훈련 코드 호로보드 분산 훈련


- 3.1.Train-TF-DDP.ipynb    
    - 세이지 메이커 TF2 DDP 분산 훈련 코드



## 참고: Amazon SageMaker Python SDK

Amazon SageMaker Python SDK는 다양한 머신러닝 및 딥러닝 프레임워크(framework)를 사용하여 Amazon SageMaker에서 모델을 쉽게 학습하고 배포할 수 있는 오픈 소스 API 및 컨테이너(containers)를 제공합니다. Amazon SageMaker Python SDK에 대한 일반적인 정보는 https://sagemaker.readthedocs.io/ 를 참조하세요.

Amazon SageMaker를 사용하여 사용자 지정 TensorFlow 코드를 사용하여 모델을 학습하고 배포할 수 있습니다. Amazon SageMaker Python SDK TensorFlow Estimator 및 model과 Amazon SageMaker 오픈 소스 TensorFlow 컨테이너를 사용하면 TensorFlow 스크립트를 작성하고 Amazon SageMaker에서 쉽게 실행할 수 있습니다.




# 참조 자료:
-Horovod 오피설 깃 리파지토리
    - https://github.com/horovod/horovod
    - tensorflow2_keras_mnist.py
        - https://github.com/horovod/horovod/blob/master/examples/tensorflow2/tensorflow2_keras_mnist.py


- Deep Learning 모델의 효과적인 분산 트레이닝과 모델 최적화 방법 - 김무현 데이터 사이언티스트(AWS)
    - https://www.youtube.com/watch?v=UFCY8YpyRkI
    

## License Summary

이 샘플 코드는 MIT-0 라이센스에 따라 제공됩니다. LICENSE 파일을 참조하십시오.


