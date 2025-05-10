# Pruning-AI: AI 기반 포도나무 가지치기 추천 시스템
**2025 쿠톤 해커톤 우수상 수상작**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/Framework-FastAPI-green)
![OpenCV](https://img.shields.io/badge/OpenCV-사용-orange)
![OpenAI](https://img.shields.io/badge/OpenAI%20API-GPT--4o-purple)

> **"AI, 모든 가지에 미래를 디자인하다!"**

저희는 **TEAM 우왕쿠왕**입니다!  
2025 KHUTHON에서 **농업의 기술화**라는 주제로 참가하여 **우수상**을 수상한 **Pruning-AI** 프로젝트의 **백엔드 저장소**입니다.

---

## 🏆 프로젝트 목표 및 성과

**복잡하고 반복적인 포도나무 가지치기 작업을 AI와 컴퓨터 비전 기술로 혁신!**

- 포도나무 이미지 입력 시, 최적 가지치기 지점 추천
- GPT-4o 기반 기대 효과 설명까지 제공하는 AI 시스템

### 주요 성과
- 🏅 2025 쿠톤 해커톤 우수상 수상  
- **FastAPI 기반 안정적 백엔드 API 구현**
- **OpenCV 기반 정밀 가지 구조 분석**
- **GPT-4o 연동으로 사용자 경험 강화**
- **독자적인 휴리스틱 알고리즘 개발**

---

## 🌟 주요 기능

### 1. `/process_dataset_image/`: 데이터셋 이미지 분석
- Buds-Dataset-main 기반 세그멘테이션 및 골격화
- Cane, Cordon, Trunk 정밀 식별 → 가지치기 지점 추천
- GPT-4o 코멘트 생성 + 시각화 이미지(Base64) 및 결과 JSON 반환

### 2. `/upload_and_analyze_live_image/`: 실시간 이미지 업로드 및 분석
- HSV 기반 세그멘테이션 → 위와 동일한 처리 과정
- 결과 이미지 및 분석 데이터 JSON 형태로 반환

### 3. `/`: API 헬스 체크

---

## 🛠️ 기술 스택

- **Backend:** Python, FastAPI, Uvicorn  
- **Image Processing:** OpenCV, NumPy  
- **AI (LLM):** OpenAI API (gpt-4o)  
- **Dataset:** Buds-Dataset-main (세그멘테이션 포함)  
- **Deployment (예시):** Cloudtype  
- **기타 라이브러리:** Pillow, python-dotenv, python-multipart

---

## 📂 프로젝트 구조

PURUNING/
├── back-end/
│ ├── main_api_heuristic.py # FastAPI 메인 애플리케이션
│ ├── cv_algorithms_heuristic.py # CV + 휴리스틱 알고리즘
│ ├── models/ # AI 모델 저장 디렉토리(현재 비어 있음)
│ ├── uploaded_images_api/ # 업로드 이미지 저장
│ ├── segmentation_results_api/ # 생성 마스크 저장
│ ├── .env.example # 환경 변수 예시
│ └── requirements.txt # 필수 라이브러리
├── Buds-Dataset-main/ # 고품질 세그멘테이션 데이터셋 (Git 제외 가능)
│ ├── Images/
│ ├── SegmentationClassPNG/
│ └── SegmentationClassVisualization/
├── .cloudtype/ # Cloudtype 배포 설정
│ └── app.yaml
├── .gitignore
└── README.md


---

## 🚀 시작 가이드

### 저장소 클론
```bash
git clone https://github.com/ssum21/Pruning_backend.git
cd Pruning_backend/back-end
```

### 가상환경 생성 및 활성화 (선택사항)
```bash
python3 -m venv venv
source venv/bin/activate 
# Windows: venv\Scripts\activate
```

### 필수 라이브러리 설치
```bash
pip install -r requirements.txt
# 만약 opencv-contrib-python이 필요하다면 (cv2.ximgproc.thinning 사용 시):
# pip install opencv-contrib-python
```

### 환경 변수 설정
```bash
OPENAI_API_KEY="sk-여기에_본인의_OpenAI_API_키를_입력하세요"
```

### FastAPI 서버 실행
```bash
uvicorn main_api_heuristic:app --reload --host 0.0.0.0 --port 8000
```

---

## 📜 라이선스
본 프로젝트는 MIT 라이선스를 따릅니다.

---

## 👨‍💻 우왕쿠왕 팀원 소개 및 역할
| 이름 | 역할 |
|------|------|
| 임수민 | 백엔드 개발, API 설계 |
| 김리원 | AI 개발, 프레젠테이션 준비 |
| 백지원 | 프론트엔드 개발(https://github.com/qorjiwon/pruning)|
| 문예빈 | UI/UX 디자인 및 프론트엔드 기여 |


