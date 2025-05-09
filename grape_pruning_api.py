import cv2
import numpy as np
import math
import os
import traceback
from typing import List, Dict, Tuple, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
# from PIL import Image # 현재 코드에서 직접 사용하지 않으므로 주석 처리 가능

# --- FastAPI 앱 초기화 ---
app = FastAPI(
    title="Grape Pruning Recommendation API",
    description="Provides pruning recommendations for grapevines based on image analysis.",
    version="0.1.0"
)

# --- 경로 설정 (사용자 파일 구조에 맞게 수정) ---
# grape_pruning_api.py 파일이 있는 디렉토리 (예: PURUNING/back-end/)
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Buds-Dataset-main 폴더는 back-end 폴더의 부모 디렉토리 아래에 위치
# PURUNING/Buds-Dataset-main/
BASE_DATASET_DIR = os.path.join(CURRENT_SCRIPT_DIR, '..', 'Buds-Dataset-main') 

IMAGES_DIR = os.path.join(BASE_DATASET_DIR, 'Images')
MASKS_DIR = os.path.join(BASE_DATASET_DIR, 'SegmentationClassPNG')

# --- 색상 정의 (BGR 순서) ---
# 중요: Buds-Dataset-main/SegmentationClassPNG 이미지에서 실제 색상값을 확인하고 정확하게 수정하세요!
# (예: 이미지 편집 프로그램의 스포이드 도구 사용)
# class_names.txt 파일이 있다면 해당 파일의 내용을 참고하여 색상과 클래스명을 매칭할 수도 있습니다.
# 아래는 예시 값입니다.
COLOR_BACKGROUND_BGR: List[int] = [0, 0, 0]      # 검은색
COLOR_CANE_BGR: List[int] = [0, 255, 0]          # 초록색 (일년생 가지, 결과모지)
COLOR_CORDON_BGR: List[int] = [0, 255, 255]      # 노란색 (2년 이상 된 수평 가지) - 예시값이므로 실제 확인!
COLOR_TRUNK_BGR: List[int] = [0, 0, 255]         # 빨간색 (주간) - 예시값이므로 실제 확인!


# --- CV 알고리즘 함수들 ---

def extract_binary_mask_from_color(segmentation_mask_bgr: np.ndarray, 
                                   target_color_bgr: List[int], 
                                   tolerance: int = 15) -> np.ndarray:
    """
    색상 기반으로 특정 객체의 이진 마스크를 추출합니다.
    """
    if not isinstance(target_color_bgr, (list, tuple)) or len(target_color_bgr) != 3:
        raise ValueError("target_color_bgr must be a list or tuple of 3 integers.")
    
    lower_bound = np.array([max(0, c - tolerance) for c in target_color_bgr], dtype=np.uint8)
    upper_bound = np.array([min(255, c + tolerance) for c in target_color_bgr], dtype=np.uint8)
    binary_mask = cv2.inRange(segmentation_mask_bgr, lower_bound, upper_bound)
    return binary_mask

def get_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """
    이진 마스크로부터 골격을 추출합니다. (OpenCV thinning 사용)
    cv2.ximgproc.thinning이 없으면 원본 마스크를 반환 (또는 에러 발생)
    """
    if not hasattr(cv2, 'ximgproc') or not hasattr(cv2.ximgproc, 'thinning'):
        print("Warning: cv2.ximgproc.thinning is not available. Please install opencv-contrib-python.")
        print("Skeletonization will be skipped, returning original binary mask.")
        # 또는 여기서 직접 구현한 zhang_suen_thinning 함수를 호출할 수 있습니다.
        # return zhang_suen_thinning_custom(binary_mask) # 직접 구현한 함수가 있다면
        return binary_mask 

    thinned = cv2.ximgproc.thinning(binary_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    return thinned

def find_virtual_buds_on_cane_skeleton(cane_skeleton_img: np.ndarray, 
                                       interval_pixels: int = 50) -> List[Dict[str, int]]:
    """
    Cane 골격 위에서 등간격으로 가상 눈 위치를 찾습니다. (단순화된 버전)
    반환: [{'x': x_coord, 'y': y_coord}, ...]
    """
    virtual_buds: List[Dict[str, int]] = []
    points_yx = np.column_stack(np.where(cane_skeleton_img > 0)) # (row, col) -> (y, x)
    
    if len(points_yx) == 0:
        return []
    
    # y, x 순으로 정렬 (위에서 아래로, 왼쪽에서 오른쪽으로) - 이 정렬이 항상 최선은 아님
    sorted_points_yx = points_yx[np.lexsort((points_yx[:, 1], points_yx[:, 0]))]

    last_added_point_xy: Optional[Tuple[int, int]] = None
    for y, x in sorted_points_yx:
        current_point_xy = (int(x), int(y))
        if not virtual_buds: 
            virtual_buds.append({'x': current_point_xy[0], 'y': current_point_xy[1]})
            last_added_point_xy = current_point_xy
        else:
            if last_added_point_xy: 
                dist = math.sqrt(
                    (current_point_xy[0] - last_added_point_xy[0])**2 + 
                    (current_point_xy[1] - last_added_point_xy[1])**2
                )
                if dist >= interval_pixels:
                    virtual_buds.append({'x': current_point_xy[0], 'y': current_point_xy[1]})
                    last_added_point_xy = current_point_xy
    return virtual_buds


def recommend_pruning_points_heuristic(virtual_buds: List[Dict[str, int]], 
                                     cane_skeleton_img: np.ndarray, 
                                     cordon_binary_mask: Optional[np.ndarray] = None, 
                                     pruning_offset_pixels: int = 25) -> List[Dict[str, int]]:
    """
    가상 눈 위치와 골격 정보를 기반으로 가지치기 지점을 추천합니다.
    """
    pruning_points: List[Dict[str, int]] = []
    if not virtual_buds:
        return []

    for bud in virtual_buds:
        bud_x, bud_y = bud['x'], bud['y']
        
        candidate_pruning_y = bud_y - pruning_offset_pixels
        candidate_pruning_x = bud_x 

        if 0 <= candidate_pruning_y < cane_skeleton_img.shape[0] and \
           0 <= candidate_pruning_x < cane_skeleton_img.shape[1] and \
           cane_skeleton_img[int(candidate_pruning_y), int(candidate_pruning_x)] > 0:
            pruning_points.append({'x': int(candidate_pruning_x), 'y': int(candidate_pruning_y)})
        else:
            pruning_points.append({'x': bud_x, 'y': bud_y, 'note': 'original_bud_location_due_to_offset_failure'})
            
    return pruning_points


# --- API 엔드포인트 ---
@app.get(
    "/get_pruning_recommendations/",
    summary="Get Pruning Recommendations for a Grapevine Image",
    response_description="A JSON object containing virtual bud locations and recommended pruning points.",
)
async def get_recommendations_endpoint(
    image_filename: str = Query(
        ..., 
        description="Filename of the original image in the 'Buds-Dataset-main/Images' folder (e.g., ZED_image_left0.png)",
        examples=["ZED_image_left0.png", "ZED_image_left1.png"]
    ),
    bud_interval_pixels: int = Query(
        50, 
        description="Approximate pixel interval between virtual buds on a cane skeleton.",
        ge=10, le=200
    ),
    pruning_offset_pixels: int = Query(
        25,
        description="Pixel offset from a virtual bud along the skeleton to recommend a pruning point.",
        ge=5, le=100
    )
):
    try:
        # 1. 파일 경로 확인 및 생성
        original_image_path = os.path.join(IMAGES_DIR, image_filename)
        segmentation_mask_path = os.path.join(MASKS_DIR, image_filename)

        if not os.path.exists(original_image_path):
            raise HTTPException(status_code=404, detail=f"Original image not found: {image_filename} at path {original_image_path}")
        if not os.path.exists(segmentation_mask_path):
            raise HTTPException(status_code=404, detail=f"Segmentation mask not found: {image_filename} at path {segmentation_mask_path}")

        # 2. 이미지 및 마스크 로드
        seg_mask_bgr = cv2.imread(segmentation_mask_path)
        if seg_mask_bgr is None:
            raise HTTPException(status_code=500, detail=f"Failed to load segmentation mask: {image_filename}")
        
        original_img_bgr = cv2.imread(original_image_path)
        if original_img_bgr is None:
            raise HTTPException(status_code=500, detail=f"Failed to load original image: {image_filename}")

        # 3. Cane 이진 마스크 추출
        cane_binary_mask = extract_binary_mask_from_color(seg_mask_bgr, COLOR_CANE_BGR)
        # cordon_binary_mask = extract_binary_mask_from_color(seg_mask_bgr, COLOR_CORDON_BGR) # 필요시 사용

        # 4. Cane 골격화
        cane_skeleton = get_skeleton(cane_binary_mask)

        # 5. 가상 눈 위치 추정
        virtual_buds = find_virtual_buds_on_cane_skeleton(cane_skeleton, interval_pixels=bud_interval_pixels)

        # 6. 가지치기 지점 추천
        pruning_points = recommend_pruning_points_heuristic(
            virtual_buds, 
            cane_skeleton, 
            # cordon_binary_mask=cordon_binary_mask, # Cordon 마스크는 현재 휴리스틱에서 미사용
            pruning_offset_pixels=pruning_offset_pixels
        )
        
        return {
            "image_filename": image_filename,
            "original_image_shape_hw": original_img_bgr.shape[:2], 
            "parameters_used": {
                "bud_interval_pixels": bud_interval_pixels,
                "pruning_offset_pixels": pruning_offset_pixels,
                "cane_color_bgr_used_for_mask": COLOR_CANE_BGR 
            },
            "virtual_buds": virtual_buds,
            "pruning_points": pruning_points,
        }

    except HTTPException as http_exc:
        raise http_exc
    except ValueError as ve: 
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Invalid input or configuration: {str(ve)}")
    except Exception as e:
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# --- 서버 실행 ---
if __name__ == "__main__":
    import uvicorn
    print(f"--- Grape Pruning API ---")
    print(f"Script location: {CURRENT_SCRIPT_DIR}")
    print(f"Attempting to use Buds-Dataset base directory: {BASE_DATASET_DIR}")
    print(f"Images directory configured as: {IMAGES_DIR}")
    print(f"Masks directory configured as: {MASKS_DIR}")
    
    if not os.path.isdir(BASE_DATASET_DIR):
        print(f"ERROR: Buds-Dataset directory not found at the expected location: {BASE_DATASET_DIR}")
        print(f"Please ensure 'Buds-Dataset-main' (or 'Buds-Dataset') folder is in the same directory as the 'back-end' folder.")
    elif not os.path.isdir(IMAGES_DIR) or not os.path.isdir(MASKS_DIR):
        print(f"ERROR: 'Images' or 'SegmentationClassPNG' subfolder not found within {BASE_DATASET_DIR}")
        print(f"Please check the structure of your 'Buds-Dataset-main' folder.")
    else:
        print(f"Dataset directories seem to be configured. Check individual file access if errors occur.")

    print(f"Used Cane BGR color for mask (verify this!): {COLOR_CANE_BGR}")
    print(f"Starting server on http://0.0.0.0:8000")
    print("Access API docs at http://localhost:8000/docs")
    
    # uvicorn grape_pruning_api:app --reload --host 0.0.0.0 --port 8000
    # 위 명령어는 터미널에서 직접 실행해주세요.
    # Python 스크립트 내에서 uvicorn.run을 사용할 때는 reload=True가 개발 중에는 유용하지만,
    # 실제 서비스 시에는 uvicorn 커맨드를 직접 사용하는 것이 일반적입니다.
    uvicorn.run("grape_pruning_api:app", host="0.0.0.0", port=8000, reload=True)