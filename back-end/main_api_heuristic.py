import os
import traceback
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np 
import math

import shutil
from fastapi import File, UploadFile, Form
from datetime import datetime
import uuid

# cv_algorithms_heuristic.py 파일에서 함수 및 변수 임포트
from cv_algorithms_heuristic import (
    extract_binary_mask_from_color,
    get_skeleton_cv,
    find_virtual_buds_heuristic,
    recommend_pruning_points_heuristic_cv,
    get_key_skeleton_points,
    COLOR_CANE_BGR,
    COLOR_CORDON_BGR,
    COLOR_TRUNK_BGR
)

app = FastAPI(
    title="포도나무 가지치기 추천 API",
    description="40개 이하의 제한된 좌표로 가지치기 추천을 제공합니다.",
    version="1.1.0"
)

# CORS 미들웨어 설정 (프론트엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 중에는 모든 오리진 허용, 프로덕션에서는 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 경로 설정 ---
# 상대 경로 사용 (Puruning 디렉토리 내부에 Buds-Dataset-main이 있는 경우)
CURRENT_API_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# '../'을 제거하고 바로 'Buds-Dataset-main'을 찾음
BASE_DATASET_DIR = os.path.join(os.path.dirname(CURRENT_API_SCRIPT_DIR), 'Buds-Dataset-main')
IMAGES_DIR = os.path.join(BASE_DATASET_DIR, 'Images')
MASKS_DIR = os.path.join(BASE_DATASET_DIR, 'SegmentationClassPNG')

# main_api_heuristic.py 파일의 경로 설정 부분 아래에 추가
# --- 업로드 디렉토리 설정 ---
UPLOAD_DIR = os.path.join(CURRENT_API_SCRIPT_DIR, 'uploaded_images')
SEGMENTATION_DIR = os.path.join(CURRENT_API_SCRIPT_DIR, 'segmentation_results')

# 디렉토리가 없으면 생성
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SEGMENTATION_DIR, exist_ok=True)

# 정적 파일 제공을 위한 설정 추가
from fastapi.staticfiles import StaticFiles
app.mount("/uploaded", StaticFiles(directory=UPLOAD_DIR), name="uploaded")
app.mount("/segmentation", StaticFiles(directory=SEGMENTATION_DIR), name="segmentation")


@app.get(
    "/get_recommendations_from_dataset/",
    summary="데이터셋 이미지에 대한 가지치기 추천 받기 (40개 이하 포인트)",
    response_description="가상 눈 위치와 추천된 가지치기 지점이 포함된 JSON 객체 (총 40개 이하 포인트).",
)
async def get_recs_from_dataset_endpoint(
    image_filename: str = Query(
        ..., 
        description="Buds-Dataset-main/Images 폴더의 파일명(예: ZED_image_left0.png)",
        examples=["ZED_image_left0.png"]
    ),
    bud_interval_pixels: int = Query(30, ge=10, le=200, description="가상 눈 사이의 픽셀 간격"),
    pruning_offset_pixels: int = Query(15, ge=5, le=100, description="가지치기 지점의 오프셋 거리(픽셀)"),
    pruning_points_count: int = Query(15, ge=1, le=30, description="추천할 가지치기 지점의 최대 개수"),
    neighborhood_radius: int = Query(30, ge=10, le=100, description="추천된 가지치기 지점 간의 최소 거리"),
    include_skeleton_points: bool = Query(True, description="골격 중요 포인트를 포함할지 여부"),
    include_virtual_buds: bool = Query(True, description="가상 눈 위치를 포함할지 여부")
):
    try:
        # 파일 경로 확인
        original_image_path = os.path.join(IMAGES_DIR, image_filename)
        segmentation_mask_path = os.path.join(MASKS_DIR, image_filename)

        if not os.path.exists(original_image_path):
            raise HTTPException(status_code=404, detail=f"원본 이미지를 찾을 수 없습니다: {image_filename}")
        if not os.path.exists(segmentation_mask_path):
            raise HTTPException(status_code=404, detail=f"세그멘테이션 마스크를 찾을 수 없습니다: {image_filename}")

        # 이미지 로딩
        seg_mask_bgr = cv2.imread(segmentation_mask_path)
        original_img_bgr = cv2.imread(original_image_path)
        
        if seg_mask_bgr is None or original_img_bgr is None:
            raise HTTPException(status_code=500, detail=f"이미지 로딩 실패: {image_filename}")

        # --- 전체 포인트 수 제한 계획 ---
        # 최대 총 포인트 수: 40개
        # 우선순위:
        # 1. 가지치기 추천 포인트: 최대 15개
        # 2. 가상 눈 위치: 최대 15개
        # 3. 골격 중요 포인트: 최대 10개
        
        # 가지치기 추천 포인트 수 (최대 15개로 제한)
        max_pruning_points = min(pruning_points_count, 15)
        
        # 가상 눈 최대 수
        max_virtual_buds = 15
        
        # 골격 중요 포인트 최대 수
        max_skeleton_points = 10
        
        # CV 처리
        cane_binary_mask = extract_binary_mask_from_color(seg_mask_bgr, COLOR_CANE_BGR)
        cane_skeleton_np = get_skeleton_cv(cane_binary_mask)
        
        # 가상 눈 위치 찾기 (최대 15개)
        virtual_buds = find_virtual_buds_heuristic(
            cane_skeleton_np, 
            interval_pixels=bud_interval_pixels,
            max_buds=max_virtual_buds
        )
        
        # 가지치기 추천 지점 계산 (최대 15개, 30픽셀 이상 떨어짐)
        pruning_points = recommend_pruning_points_heuristic_cv(
            virtual_buds, 
            cane_skeleton_np,
            pruning_offset_pixels=pruning_offset_pixels,
            max_recommendations=max_pruning_points,
            neighborhood_radius=neighborhood_radius
        )
        
        # 골격 중요 포인트 (교차점, 끝점 등) 최대 10개
        key_skeleton_points = []
        if include_skeleton_points:
            key_skeleton_points = get_key_skeleton_points(cane_skeleton_np, max_points=max_skeleton_points)
        
        # 응답에 포함할 데이터 결정
        response_virtual_buds = []
        if include_virtual_buds:
            response_virtual_buds = virtual_buds
        
        # 총 포인트 수 계산 및 로깅
        total_points = len(pruning_points) + len(response_virtual_buds) + len(key_skeleton_points)
        print(f"총 포인트 수: {total_points} (가지치기: {len(pruning_points)}, 가상 눈: {len(response_virtual_buds)}, 골격: {len(key_skeleton_points)})")
        
        # --- 응답 데이터 구성 ---
        response_data = {
            "image_filename_processed": image_filename,
            "original_image_shape_hw": [int(s) for s in original_img_bgr.shape[:2]],
            "parameters_used": {
                "bud_interval_pixels": int(bud_interval_pixels),
                "pruning_offset_pixels": int(pruning_offset_pixels),
                "max_recommendations": int(max_pruning_points),
                "neighborhood_radius": int(neighborhood_radius)
            },
            "total_points_count": total_points,
            "pruning_points": pruning_points,  # 최우선 데이터
        }
        
        # 선택적으로 추가 데이터 포함
        if include_virtual_buds:
            response_data["virtual_buds"] = response_virtual_buds
            
        if include_skeleton_points:
            response_data["key_skeleton_points"] = key_skeleton_points
            
        return JSONResponse(content=response_data)

    except HTTPException as http_exc:
        # FastAPI 예외는 그대로 전달
        raise http_exc
    except ValueError as ve: 
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"잘못된 입력 또는 설정: {str(ve)}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"내부 서버 오류 발생: {str(e)}")

async def segment_image(image_path: str) -> str:
    """
    업로드된 이미지를 세그멘테이션하여 마스크를 생성합니다.
    
    Args:
        image_path: 원본 이미지 경로
        
    Returns:
        생성된 세그멘테이션 마스크 경로
    """
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    
    # 이미지를 HSV 색상 공간으로 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 녹색 영역 추출 (Cane으로 간주)
    # HSV에서 녹색 범위: Hue 40-80, Saturation 40-255, Value 40-255
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    cane_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 노란색 영역 추출 (Cordon으로 간주)
    # HSV에서 노란색 범위: Hue 20-40, Saturation 100-255, Value 100-255
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    cordon_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 갈색/빨간색 영역 추출 (Trunk로 간주)
    # HSV에서 빨간색 범위: 
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    trunk_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    trunk_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    trunk_mask = cv2.bitwise_or(trunk_mask1, trunk_mask2)
    
    # 세그멘테이션 결과 이미지 생성 (BGR 형식)
    segmentation_result = np.zeros_like(img)
    
    # Cane(초록색)
    segmentation_result[cane_mask > 0] = COLOR_CANE_BGR
    
    # Cordon(노란색)
    segmentation_result[cordon_mask > 0] = COLOR_CORDON_BGR
    
    # Trunk(빨간색)
    segmentation_result[trunk_mask > 0] = COLOR_TRUNK_BGR
    
    # 결과 이미지 저장
    filename = os.path.basename(image_path)
    segmentation_path = os.path.join(SEGMENTATION_DIR, f"seg_{filename}")
    cv2.imwrite(segmentation_path, segmentation_result)
    
    return segmentation_path


@app.get("/", summary="API 상태 확인")
async def root():
    return {"status": "online", "message": "포도나무 가지치기 추천 API가 실행 중입니다.", "version": "1.1.0"}

@app.post(
    "/upload_image/",
    summary="새 이미지 업로드 및 분석",
    response_description="업로드된 이미지에 대한 가지치기 추천 결과",
)
async def upload_image_endpoint(
    image: UploadFile = File(..., description="업로드할 포도나무 이미지 파일"),
    bud_interval_pixels: int = Form(30, description="가상 눈 사이의 픽셀 간격"),
    pruning_offset_pixels: int = Form(15, description="가지치기 지점의 오프셋 거리(픽셀)"),
    pruning_points_count: int = Form(15, description="추천할 가지치기 지점의 최대 개수"),
    neighborhood_radius: int = Form(30, description="추천된 가지치기 지점 간의 최소 거리"),
    include_skeleton_points: bool = Form(True, description="골격 중요 포인트를 포함할지 여부"),
    include_virtual_buds: bool = Form(True, description="가상 눈 위치를 포함할지 여부")
):
    try:
        # 파일명 생성 (고유 ID + 원본 파일명)
        file_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        original_filename = image.filename
        safe_filename = f"{file_id}_{timestamp}_{original_filename}"
        safe_filename = safe_filename.replace(" ", "_")
        
        # 업로드 파일 저장 경로
        file_path = os.path.join(UPLOAD_DIR, safe_filename)
        
        # 파일 저장
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
            
        # 세그멘테이션 수행
        segmentation_mask_path = await segment_image(file_path)
        
        # 이미지 로딩
        seg_mask_bgr = cv2.imread(segmentation_mask_path)
        original_img_bgr = cv2.imread(file_path)
        
        if seg_mask_bgr is None or original_img_bgr is None:
            raise HTTPException(status_code=500, detail="이미지 처리 중 오류가 발생했습니다.")
            
        # --- 여기서부터는 기존 분석 코드와 동일 ---
        # 가지치기 추천 포인트 수 (최대 15개로 제한)
        max_pruning_points = min(pruning_points_count, 15)
        
        # 가상 눈 최대 수
        max_virtual_buds = 15
        
        # 골격 중요 포인트 최대 수
        max_skeleton_points = 10
        
        # CV 처리
        cane_binary_mask = extract_binary_mask_from_color(seg_mask_bgr, COLOR_CANE_BGR)
        cane_skeleton_np = get_skeleton_cv(cane_binary_mask)
        
        # 가상 눈 위치 찾기 (최대 15개)
        virtual_buds = find_virtual_buds_heuristic(
            cane_skeleton_np, 
            interval_pixels=bud_interval_pixels,
            max_buds=max_virtual_buds
        )
        
        # 가지치기 추천 지점 계산 (최대 15개, 30픽셀 이상 떨어짐)
        pruning_points = recommend_pruning_points_heuristic_cv(
            virtual_buds, 
            cane_skeleton_np,
            pruning_offset_pixels=pruning_offset_pixels,
            max_recommendations=max_pruning_points,
            neighborhood_radius=neighborhood_radius
        )
        
        # 골격 중요 포인트 (교차점, 끝점 등) 최대 10개
        key_skeleton_points = []
        if include_skeleton_points:
            key_skeleton_points = get_key_skeleton_points(cane_skeleton_np, max_points=max_skeleton_points)
        
        # 응답에 포함할 데이터 결정
        response_virtual_buds = []
        if include_virtual_buds:
            response_virtual_buds = virtual_buds
        
        # 총 포인트 수 계산
        total_points = len(pruning_points) + len(response_virtual_buds) + len(key_skeleton_points)
        
        # 이미지 URL 생성 (외부에서 접근 가능한 URL)
        image_url = f"/uploaded/{safe_filename}"
        segmentation_url = f"/segmentation/seg_{safe_filename}"
        
        # 응답 데이터 구성
        response_data = {
            "image_id": file_id,
            "original_filename": original_filename,
            "image_url": image_url,
            "segmentation_url": segmentation_url,
            "original_image_shape_hw": [int(s) for s in original_img_bgr.shape[:2]],
            "parameters_used": {
                "bud_interval_pixels": int(bud_interval_pixels),
                "pruning_offset_pixels": int(pruning_offset_pixels),
                "max_recommendations": int(max_pruning_points),
                "neighborhood_radius": int(neighborhood_radius)
            },
            "total_points_count": total_points,
            "pruning_points": pruning_points,
        }
        
        # 선택적으로 추가 데이터 포함
        if include_virtual_buds:
            response_data["virtual_buds"] = response_virtual_buds
            
        if include_skeleton_points:
            response_data["key_skeleton_points"] = key_skeleton_points
            
        return JSONResponse(content=response_data)
        
    except HTTPException as http_exc:
        # 임시 파일 정리
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise http_exc
    except Exception as e:
        # 임시 파일 정리
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"내부 서버 오류 발생: {str(e)}")


# --- 서버 실행 ---
if __name__ == "__main__":
    import uvicorn
    print(f"--- 포도나무 가지치기 추천 API (포인트 수 제한 버전) ---")
    print(f"스크립트 위치: {CURRENT_API_SCRIPT_DIR}")
    print(f"데이터셋 기본 디렉토리: {BASE_DATASET_DIR}")
    
    # 경로 유효성 검사
    if not os.path.isdir(BASE_DATASET_DIR) or \
       not os.path.isdir(IMAGES_DIR) or \
       not os.path.isdir(MASKS_DIR):
        print(f"오류: 데이터셋 디렉토리를 찾을 수 없거나 구성이 잘못되었습니다. 경로를 확인하세요.")
        print(f"  BASE_DATASET_DIR: {BASE_DATASET_DIR} (존재: {os.path.isdir(BASE_DATASET_DIR)})")
        print(f"  IMAGES_DIR: {IMAGES_DIR} (존재: {os.path.isdir(IMAGES_DIR)})")
        print(f"  MASKS_DIR: {MASKS_DIR} (존재: {os.path.isdir(MASKS_DIR)})")
    else:
        print(f"데이터셋 디렉토리가 정상입니다.")
    
    print(f"마스크용 Cane BGR 색상: {COLOR_CANE_BGR}")
    print(f"서버 시작: http://0.0.0.0:8000")
    print("API 문서: http://localhost:8000/docs")
    
    uvicorn.run("main_api_heuristic:app", host="0.0.0.0", port=8000, reload=True)