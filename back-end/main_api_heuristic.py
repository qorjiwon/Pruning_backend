import os
import traceback
from typing import List, Dict, Optional, Any
import base64 
import io 
from datetime import datetime
import uuid 
import shutil 

from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

import cv2
import numpy as np 
import math

# OpenAI 라이브러리 임포트
import openai 

load_dotenv() # .env 파일에서 환경 변수를 로드합니다.

# cv_algorithms_heuristic.py 파일에서 함수 및 변수 임포트
from cv_algorithms_heuristic import (
    extract_binary_mask_from_color,
    get_skeleton_cv,
    find_virtual_buds_heuristic,
    recommend_pruning_points_heuristic_cv,
    get_key_skeleton_points, # 이 함수가 cv_algorithms_heuristic.py에 정의되어 있어야 함
    COLOR_CANE_BGR,
    COLOR_CORDON_BGR, 
    COLOR_TRUNK_BGR,
    # create_visualization_image # 디버깅용 시각화 함수 (API에서는 직접 사용 안 함)
)

app = FastAPI(
    title="포도나무 가지치기 추천 API (FastAPI + ChatGPT)",
    description="데이터셋 이미지 또는 업로드된 이미지에 대해 세그멘테이션 및 휴리스틱 기반 가지치기 추천과 ChatGPT 코멘트를 제공합니다.",
    version="1.3.0" # 버전 업데이트
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 경로 설정 ---
CURRENT_API_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Buds-Dataset-main은 back-end 폴더의 부모(PURUNING) 폴더 아래에 위치
BASE_DATASET_DIR = os.path.join(os.path.dirname(CURRENT_API_SCRIPT_DIR), 'Buds-Dataset-main')
IMAGES_DIR = os.path.join(BASE_DATASET_DIR, 'Images')
MASKS_DIR = os.path.join(BASE_DATASET_DIR, 'SegmentationClassPNG')

UPLOAD_DIR_NAME = 'uploaded_images_api' 
SEGMENTATION_DIR_NAME = 'segmentation_results_api' 

UPLOAD_DIR = os.path.join(CURRENT_API_SCRIPT_DIR, UPLOAD_DIR_NAME)
SEGMENTATION_DIR = os.path.join(CURRENT_API_SCRIPT_DIR, SEGMENTATION_DIR_NAME)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SEGMENTATION_DIR, exist_ok=True)

app.mount(f"/{UPLOAD_DIR_NAME}", StaticFiles(directory=UPLOAD_DIR), name="uploaded_api_files")
app.mount(f"/{SEGMENTATION_DIR_NAME}", StaticFiles(directory=SEGMENTATION_DIR), name="segmentation_api_files")


# --- OpenAI API 키 설정 ---
try:
    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    if not client.api_key:
        print("경고: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. ChatGPT 연동이 작동하지 않을 수 있습니다.")
except Exception as e:
    print(f"OpenAI 클라이언트 초기화 중 오류: {e}")
    client = None

# --- 이미지 Base64 인코딩 함수 ---
def encode_image_to_base64_str(image_path: str, image_format: str = "jpeg") -> Optional[str]:
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/{image_format};base64,{encoded_string}"
    except Exception as e:
        print(f"이미지 Base64 인코딩 중 오류 ({image_path}): {e}")
        return None

# --- ChatGPT 호출 함수 ---
def get_chatgpt_comment_with_image(base64_image_str: Optional[str], num_pruning_points: int, image_filename: str) -> str:
    if not client or not client.api_key:
        return "OpenAI API 키가 설정되지 않아 코멘트를 생성할 수 없습니다."
    if not base64_image_str:
        return "코멘트 생성을 위한 이미지가 제공되지 않았습니다 (Base64 인코딩 실패 가능성)."
    try:
        prompt_text = (
            f"다음은 '{image_filename}' 포도나무 이미지입니다. 이 이미지에 대해 인공지능이 약 {num_pruning_points}개의 가지치기 지점을 추천했습니다. "
            "이 이미지가 포도나무인 것으로 확인된다는 것을 언급해."
            "가지치기를 통해서 어느 정도의 수확량 증가를 기대할 수 있는지를 예상 수치를 수치적으로 전망하는 코멘트를 작성하세요."
            "가지치기가 어떠한 긍정적인 영향을 미칠 것으로 기대되는지, 어떻게 가지치는 게 좋은지에 대해 조언하세요. "
        )
        messages_payload = [
            {"role": "system", "content": "당신은 농업, 특히 포도 재배에 대한 지식을 바탕으로 사용자에게 긍정적이고 희망적인 조언을 해주는 AI 어시스턴트입니다."},
            {"role": "user", "content": [{"type": "text", "text": prompt_text},
                                       {"type": "image_url", "image_url": {"url": base64_image_str, "detail": "low"}}]}
        ]
        chat_completion = client.chat.completions.create(
            messages=messages_payload, model="gpt-4o", max_tokens=200, temperature=0.7, n=1
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"ChatGPT API 호출 중 오류 (이미지 포함): {e}")
        error_detail = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                if 'error' in error_data and 'message' in error_data['error']:
                    error_detail = error_data['error']['message']
            except: pass
        return f"ChatGPT 코멘트 생성 중 오류: {error_detail}"

# --- HSV 색상 기반 세그멘테이션 함수 ---
def segment_image_using_hsv(image_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None: raise ValueError(f"이미지 로드 실패: {image_path}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masks = {}
    # HSV 범위는 실제 이미지들을 보고 조정해야 합니다.
    masks['cane'] = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))   # 초록색 범위 조정
    masks['cordon'] = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255])) # 노란색 범위 조정
    mask_r1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([160, 70, 50]), np.array([180, 255, 255])) # 빨간색 범위 조정
    masks['trunk'] = cv2.bitwise_or(mask_r1, mask_r2)
    
    segmentation_result_bgr = np.zeros_like(img)
    # 마스크 적용 시 해당 색상으로 채우기
    segmentation_result_bgr[masks['cane'] > 0] = COLOR_CANE_BGR
    segmentation_result_bgr[masks['cordon'] > 0] = COLOR_CORDON_BGR
    segmentation_result_bgr[masks['trunk'] > 0] = COLOR_TRUNK_BGR
    
    filename = os.path.basename(image_path)
    # SEGMENTATION_DIR 경로 사용
    segmentation_path = os.path.join(SEGMENTATION_DIR, f"seg_hsv_{filename}") 
    cv2.imwrite(segmentation_path, segmentation_result_bgr)
    return segmentation_path

# --- 공통 분석 로직 함수 ---
def _analyze_image_common(
    original_img_bgr: np.ndarray, 
    seg_mask_bgr: np.ndarray, # 이 함수는 세그멘테이션 마스크를 직접 받음
    bud_interval_pixels: int,
    pruning_offset_pixels: int,
    pruning_points_count: int, 
    neighborhood_radius: int,
    include_skeleton_points: bool,
    include_virtual_buds: bool
) -> Dict[str, Any]:
    
    max_pruning_points = min(pruning_points_count, 15) 
    max_virtual_buds = 20 # 가상 눈은 조금 더 많이 생성하고 pruning_points에서 필터링
    max_key_skeleton_points = 10

    cane_binary_mask = extract_binary_mask_from_color(seg_mask_bgr, COLOR_CANE_BGR)
    cane_skeleton_np = get_skeleton_cv(cane_binary_mask)
    
    virtual_buds = find_virtual_buds_heuristic(
        cane_skeleton_np, interval_pixels=bud_interval_pixels, max_buds=max_virtual_buds
    )
    pruning_points = recommend_pruning_points_heuristic_cv(
        virtual_buds, cane_skeleton_np, pruning_offset_pixels=pruning_offset_pixels,
        max_recommendations=max_pruning_points, neighborhood_radius=neighborhood_radius
    )
    
    key_skeleton_points_list = []
    if include_skeleton_points:
        key_skeleton_points_list = get_key_skeleton_points(cane_skeleton_np, max_points=max_key_skeleton_points)
    
    response_virtual_buds_list = virtual_buds if include_virtual_buds else []
    
    total_points = len(pruning_points) + len(response_virtual_buds_list) + len(key_skeleton_points_list)

    result = {
        "original_image_shape_hw": [int(s) for s in original_img_bgr.shape[:2]],
        "parameters_used": {
            "bud_interval_pixels": bud_interval_pixels,
            "pruning_offset_pixels": pruning_offset_pixels,
            "max_recommendations_requested": max_pruning_points, 
            "neighborhood_radius": neighborhood_radius
        },
        "total_points_display": total_points,
        "pruning_points": pruning_points,
    }
    if include_virtual_buds: result["virtual_buds"] = response_virtual_buds_list
    if include_skeleton_points: result["key_skeleton_points"] = key_skeleton_points_list
    return result

# --- 엔드포인트들 ---
@app.get("/", summary="API 상태 확인")
async def root():
    return {"status": "online", "message": "포도나무 가지치기 추천 API가 실행 중입니다.", "version": "1.3.0"}

@app.get(
    "/get_recommendations_from_dataset/",
    summary="데이터셋 이미지에 대한 가지치기 추천 및 ChatGPT 코멘트 받기",
)
async def get_recs_from_dataset_with_chatgpt_endpoint(
    image_filename: str = Query(..., examples=["ZED_image_left0.png"]),
    bud_interval_pixels: int = Query(30, ge=10, le=200),
    pruning_offset_pixels: int = Query(15, ge=5, le=100),
    pruning_points_count: int = Query(15, ge=1, le=30),
    neighborhood_radius: int = Query(30, ge=10, le=100),
    include_skeleton_points: bool = Query(True),
    include_virtual_buds: bool = Query(True)
):
    try:
        original_image_path = os.path.join(IMAGES_DIR, image_filename)
        segmentation_mask_path = os.path.join(MASKS_DIR, image_filename)

        if not os.path.exists(original_image_path):
            raise HTTPException(status_code=404, detail=f"Original image not found: {image_filename} at {original_image_path}")
        if not os.path.exists(segmentation_mask_path):
            raise HTTPException(status_code=404, detail=f"Segmentation mask not found: {image_filename} at {segmentation_mask_path}")

        seg_mask_bgr = cv2.imread(segmentation_mask_path)
        original_img_bgr = cv2.imread(original_image_path)
        if seg_mask_bgr is None or original_img_bgr is None:
            raise HTTPException(status_code=500, detail=f"Failed to load image or mask for {image_filename}")

        analysis_result = _analyze_image_common(
            original_img_bgr, seg_mask_bgr, bud_interval_pixels, pruning_offset_pixels,
            pruning_points_count, neighborhood_radius, include_skeleton_points, include_virtual_buds
        )
        
        image_format = "png" if image_filename.lower().endswith(".png") else "jpeg"
        base64_image_data = encode_image_to_base64_str(original_image_path, image_format=image_format)
        chatgpt_comment = get_chatgpt_comment_with_image(base64_image_data, len(analysis_result.get("pruning_points", [])), image_filename) 
        
        response_data = {
            "image_filename_processed": image_filename,
            **analysis_result, 
            "chatgpt_comment": chatgpt_comment,
        }
        return JSONResponse(content=response_data)
    except HTTPException as http_exc: raise http_exc
    except ValueError as ve: traceback.print_exc(); raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
    except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.post(
    "/upload_image_and_get_recommendations/",
    summary="Upload a new image, segment it (HSV based), and get recommendations with ChatGPT comment.",
)
async def upload_image_and_analyze_endpoint(
    imageFile: UploadFile = File(..., description="Grapevine image file to upload and analyze."),
    bud_interval_pixels: int = Form(30),
    pruning_offset_pixels: int = Form(15),
    pruning_points_count: int = Form(15),
    neighborhood_radius: int = Form(30),
    include_skeleton_points: bool = Form(True),
    include_virtual_buds: bool = Form(True)
):
    temp_file_path: Optional[str] = None
    temp_seg_mask_path: Optional[str] = None
    try:
        file_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        original_filename = imageFile.filename if imageFile.filename else "uploaded_image.tmp"
        base, ext = os.path.splitext(original_filename)
        ext = ext.lower() if ext else ".tmp" # 확장자 소문자 통일
        # 지원하는 이미지 확장자인지 간단히 확인 (선택 사항)
        supported_extensions = ['.png', '.jpg', '.jpeg']
        if ext not in supported_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Please upload PNG, JPG, or JPEG.")

        safe_filename = f"{file_id}_{timestamp}_{base}{ext}".replace(" ", "_").replace("..", "_")
        temp_file_path = os.path.join(UPLOAD_DIR, safe_filename)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(imageFile.file, buffer)
            
        temp_seg_mask_path = segment_image_using_hsv(temp_file_path) 
        
        seg_mask_bgr = cv2.imread(temp_seg_mask_path)
        original_img_bgr = cv2.imread(temp_file_path)
        
        if seg_mask_bgr is None or original_img_bgr is None:
            raise HTTPException(status_code=500, detail="Uploaded image processing failed (imread).")

        analysis_result = _analyze_image_common(
            original_img_bgr, seg_mask_bgr, bud_interval_pixels, pruning_offset_pixels,
            pruning_points_count, neighborhood_radius, include_skeleton_points, include_virtual_buds
        )
        
        image_format_for_base64 = ext[1:] if ext and len(ext) > 1 else "jpeg" 
        base64_image_data = encode_image_to_base64_str(temp_file_path, image_format=image_format_for_base64)
        chatgpt_comment = get_chatgpt_comment_with_image(base64_image_data, len(analysis_result.get("pruning_points", [])), original_filename)
        
        response_data = {
            "image_id": file_id,
            "original_filename": original_filename,
            "processed_image_url": f"/{UPLOAD_DIR_NAME}/{safe_filename}",
            "generated_segmentation_mask_url": f"/{SEGMENTATION_DIR_NAME}/{os.path.basename(temp_seg_mask_path)}",
            **analysis_result,
            "chatgpt_comment": chatgpt_comment,
        }
        return JSONResponse(content=response_data)
        
    except HTTPException as http_exc: 
        _cleanup_temp_files(temp_file_path, temp_seg_mask_path)
        raise http_exc
    except Exception as e: 
        _cleanup_temp_files(temp_file_path, temp_seg_mask_path)
        traceback.print_exc(); 
        raise HTTPException(status_code=500, detail=f"Server error during upload: {str(e)}")
    finally: 
        if imageFile and hasattr(imageFile.file, 'close') and not imageFile.file.closed: 
            imageFile.file.close()

def _cleanup_temp_files(file_path: Optional[str], seg_mask_path: Optional[str]):
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up uploaded file: {file_path}")
    except Exception as e:
        print(f"Error cleaning up original uploaded file {file_path}: {e}")
    try:
        if seg_mask_path and os.path.exists(seg_mask_path):
            os.remove(seg_mask_path)
            print(f"Cleaned up generated segmentation mask: {seg_mask_path}")
    except Exception as e:
        print(f"Error cleaning up generated segmentation mask {seg_mask_path}: {e}")

# --- 서버 실행 ---
"""
if __name__ == "__main__":
    import uvicorn
    print(f"--- FastAPI Grape Pruning API with Image Comment (ChatGPT) ---")
    print(f"Script location: {CURRENT_API_SCRIPT_DIR}")
    print(f"Attempting to use Buds-Dataset base directory: {BASE_DATASET_DIR}")
    if not os.path.isdir(BASE_DATASET_DIR) or \
       not os.path.isdir(IMAGES_DIR) or \
       not os.path.isdir(MASKS_DIR):
        print(f"ERROR: Dataset directories not found or misconfigured. Check paths.")
    else:
        print(f"Dataset directories OK.")
    print(f"Uploaded images will be saved to: {UPLOAD_DIR}")
    print(f"Generated segmentation masks (from upload) will be saved to: {SEGMENTATION_DIR}")
    print(f"Used Cane BGR color for mask (verify!): {COLOR_CANE_BGR}")
    print(f"Starting server on http://0.0.0.0:8000")
    print("Access API docs at http://localhost:8000/docs")
    
    uvicorn.run("main_api_heuristic:app", host="0.0.0.0", port=8000, reload=True)
"""