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
from dotenv import load_dotenv # .env 파일 로드용

import cv2
import numpy as np 
import math

# OpenAI 라이브러리 임포트
import openai 

load_dotenv() # 스크립트 시작 시 .env 파일에서 환경 변수 로드

# cv_algorithms_heuristic.py 파일에서 함수 및 변수 임포트
# 이 함수들이 반환하는 좌표는 이미 파이썬 기본 int 타입이라고 가정합니다.
from cv_algorithms_heuristic import (
    extract_binary_mask_from_color,
    get_skeleton_cv,
    find_virtual_buds_heuristic,
    recommend_pruning_points_heuristic_cv,
    get_key_skeleton_points,
    draw_analysis_on_image, # 시각화 함수
    COLOR_CANE_BGR,
    COLOR_CORDON_BGR, 
    COLOR_TRUNK_BGR  
)

app = FastAPI(
    title="포도나무 가지치기 추천 API (FastAPI + ChatGPT)",
    description="데이터셋 이미지 또는 업로드된 이미지에 대해 세그멘테이션 및 휴리스틱 기반 가지치기 추천과 ChatGPT 코멘트를 제공합니다.",
    version="1.3.2" # 버전 업데이트
)

# CORS 미들웨어 설정
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- 경로 설정 ---
CURRENT_API_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # back-end 폴더를 가리킴

# Buds-Dataset-main 폴더가 이제 CURRENT_API_SCRIPT_DIR (back-end 폴더) 내부에 있으므로,
# 바로 해당 폴더 이름을 join합니다.
BASE_DATASET_DIR = os.path.join(CURRENT_API_SCRIPT_DIR, 'Buds-Dataset-main')
IMAGES_DIR = os.path.join(BASE_DATASET_DIR, 'Images')
MASKS_DIR = os.path.join(BASE_DATASET_DIR, 'SegmentationClassPNG')
VISUALIZATION_DIR = os.path.join(BASE_DATASET_DIR, 'SegmentationClassVisualization')

UPLOAD_DIR_NAME = 'uploaded_images_api' 
SEGMENTATION_DIR_NAME = 'segmentation_results_api' 
UPLOAD_DIR = os.path.join(CURRENT_API_SCRIPT_DIR, UPLOAD_DIR_NAME)
SEGMENTATION_DIR = os.path.join(CURRENT_API_SCRIPT_DIR, SEGMENTATION_DIR_NAME)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SEGMENTATION_DIR, exist_ok=True)
app.mount(f"/{UPLOAD_DIR_NAME}", StaticFiles(directory=UPLOAD_DIR), name="uploaded_api_files")
app.mount(f"/{SEGMENTATION_DIR_NAME}", StaticFiles(directory=SEGMENTATION_DIR), name="segmentation_api_files")

def find_most_similar_dataset_image(
    uploaded_image_bgr: np.ndarray,
    dataset_images_dir: str = IMAGES_DIR,
    visualization_dir: str = VISUALIZATION_DIR
) -> Optional[tuple[str, str, str]]: # (matched_original_path, matched_visualization_path, matched_base_filename)
    """
    업로드된 이미지와 가장 유사한 데이터셋 이미지를 찾고,
    해당하는 원본 경로, 시각화 경로, 기본 파일명을 반환합니다.
    """
    if uploaded_image_bgr is None:
        return None

    uploaded_hist = cv2.calcHist([uploaded_image_bgr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(uploaded_hist, uploaded_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    best_match_score = -1  # HISTCMP_CORREL은 높을수록 유사
    matched_original_path: Optional[str] = None
    matched_visualization_path: Optional[str] = None
    matched_base_filename: Optional[str] = None

    if not os.path.exists(dataset_images_dir):
        print(f"Warning: Dataset images directory not found: {dataset_images_dir}")
        return None

    for filename in os.listdir(dataset_images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            dataset_image_path = os.path.join(dataset_images_dir, filename)
            dataset_img_bgr = cv2.imread(dataset_image_path)

            if dataset_img_bgr is None:
                continue

            current_hist = cv2.calcHist([dataset_img_bgr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(current_hist, current_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            # 히스토그램 비교 (상관관계 방법 사용, 값이 1에 가까울수록 유사)
            score = cv2.compareHist(uploaded_hist, current_hist, cv2.HISTCMP_CORREL)

            if score > best_match_score:
                best_match_score = score
                base_filename, _ = os.path.splitext(filename)

                # 해당 visualization 이미지 경로 찾기 (jpg 또는 png)
                vis_path_jpg = os.path.join(visualization_dir, f"{base_filename}.jpg")
                vis_path_png = os.path.join(visualization_dir, f"{base_filename}.png")

                current_vis_path = None
                if os.path.exists(vis_path_jpg):
                    current_vis_path = vis_path_jpg
                elif os.path.exists(vis_path_png):
                    current_vis_path = vis_path_png

                if current_vis_path: # Visualization 이미지가 있는 경우에만 매칭 대상으로 고려
                    matched_original_path = dataset_image_path
                    matched_visualization_path = current_vis_path
                    matched_base_filename = base_filename


    # 일정 유사도 이상일 때만 매칭된 것으로 간주 (임계값은 실험을 통해 조정)
    similarity_threshold = 0.5 # 예시 임계값
    if best_match_score >= similarity_threshold and matched_visualization_path:
        print(f"Found similar dataset image: {matched_base_filename} with score: {best_match_score}")
        return matched_original_path, matched_visualization_path, matched_base_filename
    else:
        print(f"No sufficiently similar dataset image found (best score: {best_match_score} for {matched_base_filename if matched_base_filename else 'N/A'}). Using uploaded image for visualization.")
        return None

# --- OpenAI API 키 설정 ---
try:
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not client.api_key: print("경고: OPENAI_API_KEY 환경 변수가 .env 파일 또는 시스템에 설정되지 않았습니다.")
except Exception as e: print(f"OpenAI 클라이언트 초기화 중 오류: {e}"); client = None

# --- 이미지 Base64 인코딩 함수 ---
def encode_image_to_base64_str(image_data: Any, image_format: str = "png", is_numpy_array: bool = False) -> Optional[str]:
    try:
        if is_numpy_array:
            if not isinstance(image_data, np.ndarray): raise ValueError("Numpy array expected for is_numpy_array=True.")
            success, buffer = cv2.imencode(f'.{image_format.lower()}', image_data)
            if not success: raise ValueError(f"cv2.imencode failed for format {image_format}")
            encoded_string = base64.b64encode(buffer).decode('utf-8')
        else: 
            if not isinstance(image_data, str) or not os.path.exists(image_data):
                raise ValueError(f"File path does not exist or not a string: {image_data}")
            with open(image_data, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/{image_format.lower()};base64,{encoded_string}"
    except Exception as e: print(f"Base64 인코딩 오류 (type: {type(image_data)}): {e}"); return None

# --- ChatGPT 호출 함수 ---
def get_chatgpt_comment_with_image(base64_image_str: Optional[str], num_pruning_points: int, image_filename: str) -> str:
    if not client or not client.api_key: return "OpenAI API 키가 설정되지 않아 코멘트를 생성할 수 없습니다."
    if not base64_image_str: return "코멘트 생성을 위한 이미지가 제공되지 않았습니다 (Base64 인코딩 실패 가능성)."
    try:
        prompt_text = (
            f"다음은 '{image_filename}' 포도나무 이미지입니다. 이 이미지에 대해 인공지능이 약 {num_pruning_points}개의 가지치기 지점을 추천했습니다. "
            "이 이미지와 추천된 가지치기 작업이 포도나무의 건강 증진, 미래의 포도송이 크기 및 당도 향상, 그리고 전반적인 수확량 증대에 "
            "어떤 긍정적인 영향을 미칠 것으로 기대되는지, 농부에게 희망을 주고 이해하기 쉬운 낙관적인 코멘트를 1~2문장으로 작성해주세요. "
            "전문적인 수치 예측보다는 일반적인 기대 효과와 격려의 메시지에 초점을 맞춰주세요."
        )
        messages_payload = [
            {"role": "system", "content": "당신은 농업, 특히 포도 재배에 대한 지식을 바탕으로 사용자에게 긍정적이고 희망적인 조언을 해주는 AI 어시스턴트입니다."},
            {"role": "user", "content": [{"type": "text", "text": prompt_text},
                                       {"type": "image_url", "image_url": {"url": base64_image_str, "detail": "low"}}]}
        ]
        chat_completion = client.chat.completions.create(
            messages=messages_payload, model="gpt-4o-mini", max_tokens=185, temperature=0.75, n=1
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"ChatGPT API 호출 중 오류 (이미지 포함): {e}")
        error_detail = str(e)
        if hasattr(e, 'response') and e.response is not None and hasattr(e.response, 'json'): # defensive coding
            try:
                error_data = e.response.json()
                if isinstance(error_data, dict) and 'error' in error_data and isinstance(error_data['error'], dict) and 'message' in error_data['error']:
                    error_detail = error_data['error']['message']
            except: pass # JSON 파싱 실패 시 원래 에러 메시지 사용
        return f"ChatGPT 코멘트 생성 중 오류: {error_detail}"

# --- HSV 색상 기반 세그멘테이션 함수 ---
def segment_image_using_hsv(image_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None: raise ValueError(f"이미지 로드 실패: {image_path}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masks = {}
    masks['cane'] = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    masks['cordon'] = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
    mask_r1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([160, 70, 50]), np.array([180, 255, 255]))
    masks['trunk'] = cv2.bitwise_or(mask_r1, mask_r2)
    segmentation_result_bgr = np.zeros_like(img)
    segmentation_result_bgr[masks['cane'] > 0] = COLOR_CANE_BGR
    segmentation_result_bgr[masks['cordon'] > 0] = COLOR_CORDON_BGR
    segmentation_result_bgr[masks['trunk'] > 0] = COLOR_TRUNK_BGR
    filename = os.path.basename(image_path)
    segmentation_path = os.path.join(SEGMENTATION_DIR, f"seg_hsv_{filename}") 
    cv2.imwrite(segmentation_path, segmentation_result_bgr)
    return segmentation_path

# --- 공통 분석 로직 함수 ---
def _analyze_image_core_logic(
    original_img_bgr: np.ndarray, 
    seg_mask_bgr: np.ndarray,
    bud_interval_pixels: int,
    pruning_offset_pixels: int,
    pruning_points_count: int, 
    neighborhood_radius: int,
    include_skeleton_points: bool,
    include_virtual_buds: bool
) -> Dict[str, Any]:
    max_pruning_points = min(pruning_points_count, 15) 
    max_virtual_buds = 20 
    max_key_skeleton_points = 10

    cane_binary_mask = extract_binary_mask_from_color(seg_mask_bgr, COLOR_CANE_BGR)
    cane_skeleton_np = get_skeleton_cv(cane_binary_mask)
    
    virtual_buds = find_virtual_buds_heuristic(
        cane_skeleton_np, interval_pixels=bud_interval_pixels, max_buds=max_virtual_buds
    )
    # recommend_pruning_points_heuristic_cv는 이미 내부적으로 int 변환된 좌표의 딕셔너리 리스트를 반환함
    pruning_points = recommend_pruning_points_heuristic_cv(
        virtual_buds, cane_skeleton_np, pruning_offset_pixels=pruning_offset_pixels,
        max_recommendations=max_pruning_points, neighborhood_radius=neighborhood_radius
    )
    
    key_skeleton_points_list = []
    if include_skeleton_points:
        # get_key_skeleton_points도 내부적으로 int 변환된 좌표의 딕셔너리 리스트를 반환함
        key_skeleton_points_list = get_key_skeleton_points(cane_skeleton_np, max_points=max_key_skeleton_points)
    
    response_virtual_buds_list = virtual_buds if include_virtual_buds else []
    
    total_points = len(pruning_points) + len(response_virtual_buds_list) + len(key_skeleton_points_list)

    # 반환되는 모든 숫자 값들이 파이썬 기본 타입인지 확인 (numpy 타입 방지)
    result = {
        "original_image_shape_hw": [int(s) for s in original_img_bgr.shape[:2]],
        "parameters_used": {
            "bud_interval_pixels": int(bud_interval_pixels),
            "pruning_offset_pixels": int(pruning_offset_pixels),
            "max_recommendations_requested": int(max_pruning_points), 
            "neighborhood_radius": int(neighborhood_radius)
        },
        "total_points_display": int(total_points),
        "pruning_points": pruning_points, # cv_algorithms에서 이미 처리됨
    }
    if include_virtual_buds: result["virtual_buds"] = response_virtual_buds_list # cv_algorithms에서 이미 처리됨
    if include_skeleton_points: result["key_skeleton_points"] = key_skeleton_points_list # cv_algorithms에서 이미 처리됨
    
    # 시각화 함수에 전달할 골격 마스크 (NumPy 배열) - API 응답에는 포함 안 함
    result["_internal_cane_skeleton_for_drawing"] = cane_skeleton_np 
    return result

# --- 엔드포인트들 ---
@app.get("/", summary="API 상태 확인")
async def root():
    return {"status": "online", "message": "포도나무 가지치기 추천 API가 실행 중입니다.", "version": "1.3.1"}

@app.get(
    "/process_dataset_image/", 
    summary="데이터셋 이미지 분석 및 시각화된 이미지, ChatGPT 코멘트 반환",
)
async def process_dataset_image_endpoint(
    image_base_filename: str = Query(..., description="Base filename (e.g., 'ZED_image_left0')", examples=["ZED_image_left0"]),
    bud_interval_pixels: int = Query(30, ge=10, le=200),
    pruning_offset_pixels: int = Query(15, ge=5, le=100),
    pruning_points_count: int = Query(15, ge=1, le=30),
    neighborhood_radius: int = Query(30, ge=10, le=100),
    draw_skeleton_on_visualization: bool = Query(True),
    draw_buds_on_visualization: bool = Query(True),
    draw_keypoints_on_visualization: bool = Query(False)
):
    try:
        original_image_filename = f"{image_base_filename}.png"
        mask_filename = f"{image_base_filename}.png"
        vis_bg_filename_jpg = f"{image_base_filename}.jpg"
        vis_bg_filename_png = f"{image_base_filename}.png"

        original_image_path = os.path.join(IMAGES_DIR, original_image_filename)
        segmentation_mask_path = os.path.join(MASKS_DIR, mask_filename)
        visualization_bg_path = os.path.join(VISUALIZATION_DIR, vis_bg_filename_jpg)
        if not os.path.exists(visualization_bg_path):
            visualization_bg_path = os.path.join(VISUALIZATION_DIR, vis_bg_filename_png)

        for p, name in [(original_image_path, "Original image"), 
                        (segmentation_mask_path, "Segmentation mask"), 
                        (visualization_bg_path, "Visualization background")]:
            if not os.path.exists(p):
                raise HTTPException(status_code=404, detail=f"{name} not found for base: {image_base_filename} at {p}")

        seg_mask_bgr = cv2.imread(segmentation_mask_path)
        visualization_bg_bgr = cv2.imread(visualization_bg_path) 
        original_img_for_gpt_bgr = cv2.imread(original_image_path)

        if seg_mask_bgr is None or visualization_bg_bgr is None or original_img_for_gpt_bgr is None:
            raise HTTPException(status_code=500, detail=f"Failed to load one or more images for {image_base_filename}")

        analysis_data = _analyze_image_core_logic(
            original_img_for_gpt_bgr, seg_mask_bgr, bud_interval_pixels, pruning_offset_pixels,
            pruning_points_count, neighborhood_radius, 
            draw_keypoints_on_visualization, draw_buds_on_visualization
        )
        
        final_visualized_image_np = draw_analysis_on_image(
            original_bgr_image=visualization_bg_bgr.copy(),
            pruning_points=analysis_data["pruning_points"],
            # "virtual_buds_for_analysis" 대신 "virtual_buds" 사용
            virtual_buds=analysis_data.get("virtual_buds") if draw_buds_on_visualization else None,
            # "key_skeleton_points_for_analysis" 대신 "key_skeleton_points" 사용
            key_skeleton_points=analysis_data.get("key_skeleton_points") if draw_keypoints_on_visualization else None,
            # _internal_cane_skeleton_for_drawing 키도 .get()으로 안전하게 접근하는 것이 좋음
            cane_skeleton_mask=analysis_data.get("_internal_cane_skeleton_for_drawing") if draw_skeleton_on_visualization else None,
            neighborhood_radius_for_pruning=neighborhood_radius
        )
        
        vis_image_format = "jpg" if visualization_bg_path.lower().endswith(".jpg") else "png"
        base64_final_visualized_image = encode_image_to_base64_str(final_visualized_image_np, image_format=vis_image_format, is_numpy_array=True)

        orig_img_format_gpt = "png" if original_image_filename.lower().endswith(".png") else "jpeg"
        base64_original_for_gpt = encode_image_to_base64_str(original_image_path, image_format=orig_img_format_gpt)
        chatgpt_comment = get_chatgpt_comment_with_image(base64_original_for_gpt, len(analysis_data.get("pruning_points", [])), original_image_filename)
        
        # _internal_cane_skeleton_for_drawing은 응답에서 제외
        if "_internal_cane_skeleton_for_drawing" in analysis_data:
            del analysis_data["_internal_cane_skeleton_for_drawing"]

        return JSONResponse(content={
            "matched_base_filename": image_base_filename,
            **analysis_data, 
            "visualized_image_base64": base64_final_visualized_image,
            "chatgpt_comment": chatgpt_comment,
        })
    # ... (에러 처리) ...
    except HTTPException as http_exc: raise http_exc
    except ValueError as ve: traceback.print_exc(); raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
    except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.post(
    "/upload_and_analyze_live_image/",
    summary="Upload live image, segment (HSV), draw results on original or similar dataset viz, get ChatGPT comment.", # 요약 업데이트
)
async def upload_live_image_and_analyze_endpoint(
    imageFile: UploadFile = File(...),
    bud_interval_pixels: int = Form(30),
    pruning_offset_pixels: int = Form(15),
    pruning_points_count: int = Form(15),
    neighborhood_radius: int = Form(30),
    draw_skeleton_on_visualization: bool = Form(True),
    draw_buds_on_visualization: bool = Form(True),
    draw_keypoints_on_visualization: bool = Form(False),
    use_dataset_visualization_if_similar: bool = Form(True) # 새로운 파라미터 추가
):
    temp_file_path: Optional[str] = None
    temp_seg_mask_path: Optional[str] = None
    try:
        # ... (파일 저장 로직은 동일)
        file_id = str(uuid.uuid4())[:8]; timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        original_filename = imageFile.filename if imageFile.filename else "live_image.tmp"
        base, ext = os.path.splitext(original_filename); ext = ext.lower() if ext else ".tmp"
        if ext not in ['.png', '.jpg', '.jpeg']: raise HTTPException(status_code=400, detail="Unsupported file type.")
        safe_filename = f"live_{file_id}_{timestamp}_{base}{ext}".replace(" ", "_").replace("..", "_")
        temp_file_path = os.path.join(UPLOAD_DIR, safe_filename)
        with open(temp_file_path, "wb") as buffer: shutil.copyfileobj(imageFile.file, buffer)

        temp_seg_mask_path = segment_image_using_hsv(temp_file_path)

        seg_mask_bgr = cv2.imread(temp_seg_mask_path)
        original_img_bgr = cv2.imread(temp_file_path) # 업로드된 원본 이미지

        if seg_mask_bgr is None or original_img_bgr is None:
            raise HTTPException(status_code=500, detail="Uploaded image processing failed (imread).")

        # --- 핵심 로직: 분석 및 시각화 배경 결정 ---
        analysis_data = _analyze_image_core_logic(
            original_img_bgr, seg_mask_bgr, bud_interval_pixels, pruning_offset_pixels,
            pruning_points_count, neighborhood_radius,
            draw_keypoints_on_visualization, draw_buds_on_visualization
        )

        visualization_background_bgr = original_img_bgr.copy() # 기본값: 업로드된 이미지
        matched_dataset_info_str = "Uploaded image used for visualization base."
        final_vis_image_format = ext[1:] if ext and len(ext) > 1 else "jpeg" # 기본 포맷

        if use_dataset_visualization_if_similar:
            similar_image_info = find_most_similar_dataset_image(original_img_bgr)
            if similar_image_info:
                _, matched_vis_path, matched_base_name = similar_image_info
                # 매칭된 SegmentationClassVisualization 이미지 로드
                matched_vis_bgr = cv2.imread(matched_vis_path)
                if matched_vis_bgr is not None:
                    visualization_background_bgr = matched_vis_bgr.copy()
                    matched_dataset_info_str = f"Used dataset visualization: {os.path.basename(matched_vis_path)} (matched with {matched_base_name})"
                    # 매칭된 시각화 이미지의 확장자 사용
                    _, vis_ext = os.path.splitext(matched_vis_path)
                    if vis_ext and len(vis_ext) > 1:
                        final_vis_image_format = vis_ext[1:].lower()
                else:
                    matched_dataset_info_str = f"Matched dataset viz ({os.path.basename(matched_vis_path)}) but failed to load. Using uploaded image."


        final_visualized_image_np = draw_analysis_on_image(
            original_bgr_image=visualization_background_bgr, # 결정된 배경 이미지 사용
            virtual_buds=analysis_data.get("virtual_buds") if draw_buds_on_visualization else None,
            pruning_points=analysis_data["pruning_points"],
            key_skeleton_points=analysis_data.get("key_skeleton_points") if draw_keypoints_on_visualization else None,
            cane_skeleton_mask=analysis_data.get("_internal_cane_skeleton_for_drawing") if draw_skeleton_on_visualization else None,
            neighborhood_radius_for_pruning=neighborhood_radius
        )

        base64_final_visualized_image = encode_image_to_base64_str(
            final_visualized_image_np,
            image_format=final_vis_image_format, # 결정된 포맷 사용
            is_numpy_array=True
        )

        # ChatGPT 코멘트용 이미지는 항상 업로드된 '원본' 이미지를 사용
        base64_original_for_gpt = encode_image_to_base64_str(temp_file_path, image_format=(ext[1:] if ext and len(ext) > 1 else "jpeg"))
        chatgpt_comment = get_chatgpt_comment_with_image(base64_original_for_gpt, len(analysis_data.get("pruning_points", [])), original_filename)

        if "_internal_cane_skeleton_for_drawing" in analysis_data:
            del analysis_data["_internal_cane_skeleton_for_drawing"]

        return JSONResponse(content={
            "image_id": file_id, "original_filename": original_filename,
            "processed_image_url": f"/{UPLOAD_DIR_NAME}/{safe_filename}",
            "generated_segmentation_mask_url": f"/{SEGMENTATION_DIR_NAME}/{os.path.basename(temp_seg_mask_path)}",
            **analysis_data,
            "visualization_info": matched_dataset_info_str, # 시각화 배경 정보 추가
            "visualized_image_base64": base64_final_visualized_image,
            "chatgpt_comment": chatgpt_comment,
        })

    except HTTPException as http_exc: _cleanup_temp_files(temp_file_path, temp_seg_mask_path); raise http_exc
    except Exception as e: _cleanup_temp_files(temp_file_path, temp_seg_mask_path); traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
    finally:
        if imageFile and hasattr(imageFile.file, 'close') and not imageFile.file.closed:
            imageFile.file.close()

def _cleanup_temp_files(file_path: Optional[str], seg_mask_path: Optional[str]):
    try:
        if file_path and os.path.exists(file_path): os.remove(file_path); print(f"Cleaned: {file_path}")
        if seg_mask_path and os.path.exists(seg_mask_path): os.remove(seg_mask_path); print(f"Cleaned: {seg_mask_path}")
    except Exception as e: print(f"Error cleaning temp files: {e}")

# --- 서버 실행 ---
if __name__ == "__main__":
    import uvicorn
    print(f"--- FastAPI Grape Pruning API with Image Comment (ChatGPT) ---")
    uvicorn.run("main_api_heuristic:app", host="0.0.0.0", port=8000, reload=True)