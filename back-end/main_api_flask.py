import os
import traceback
from flask import Flask, request, jsonify
import cv2
import numpy as np
import math # cv_algorithms_heuristic.py에서 math를 사용하므로

# cv_algorithms_heuristic.py 파일에서 함수 및 변수 임포트
# Flask에서는 같은 디렉토리에 있다면 바로 임포트 가능
from cv_algorithms_heuristic import (
    extract_binary_mask_from_color,
    get_skeleton_cv,
    find_virtual_buds_heuristic,
    recommend_pruning_points_heuristic_cv,
    COLOR_CANE_BGR
)

app = Flask(__name__)

# --- 경로 설정 (파일 구조에 맞게) ---
# 이 Flask 스크립트(main_api_flask.py)가 있는 디렉토리
CURRENT_FLASK_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Buds-Dataset-main 폴더는 이 스크립트의 부모 폴더(PURUNING) 아래에 위치
BASE_DATASET_DIR = os.path.join(CURRENT_FLASK_SCRIPT_DIR, '..', 'Buds-Dataset-main')
IMAGES_DIR = os.path.join(BASE_DATASET_DIR, 'Images')
MASKS_DIR = os.path.join(BASE_DATASET_DIR, 'SegmentationClassPNG')

@app.route('/get_recommendations_flask', methods=['GET'])
def get_recommendations_flask_endpoint():
    try:
        # 쿼리 파라미터 가져오기
        image_filename = request.args.get('image_filename')
        bud_interval_pixels_str = request.args.get('bud_interval_pixels', '50') # 기본값 50
        pruning_offset_pixels_str = request.args.get('pruning_offset_pixels', '25') # 기본값 25

        if not image_filename:
            return jsonify({"error": "image_filename parameter is required"}), 400

        try:
            bud_interval_pixels = int(bud_interval_pixels_str)
            pruning_offset_pixels = int(pruning_offset_pixels_str)
            if not (10 <= bud_interval_pixels <= 200 and 5 <= pruning_offset_pixels <= 100):
                raise ValueError("Parameter value out of range.")
        except ValueError:
            return jsonify({"error": "Invalid integer value for bud_interval_pixels or pruning_offset_pixels, or value out of range."}), 400


        # --- 디버깅용 경로 출력 ---
        print(f"--- Flask Request for image_filename: {image_filename} ---")
        print(f"BASE_DATASET_DIR: {BASE_DATASET_DIR}")
        print(f"IMAGES_DIR: {IMAGES_DIR}")
        # --- 경로 확인 ---
        original_image_path = os.path.join(IMAGES_DIR, image_filename)
        segmentation_mask_path = os.path.join(MASKS_DIR, image_filename)
        print(f"Attempting to load original image from: {original_image_path}")


        if not os.path.exists(original_image_path):
            return jsonify({"error": f"Original image not found: {image_filename} at {original_image_path}"}), 404
        if not os.path.exists(segmentation_mask_path):
            return jsonify({"error": f"Segmentation mask not found: {image_filename} at {segmentation_mask_path}"}), 404

        seg_mask_bgr = cv2.imread(segmentation_mask_path)
        original_img_bgr = cv2.imread(original_image_path)
        
        if seg_mask_bgr is None or original_img_bgr is None:
            return jsonify({"error": f"Failed to load image or mask for {image_filename}"}), 500

        # --- CV 처리 (cv_algorithms_heuristic.py 함수 사용) ---
        cane_binary_mask = extract_binary_mask_from_color(seg_mask_bgr, COLOR_CANE_BGR)
        cane_skeleton_np = get_skeleton_cv(cane_binary_mask)
        virtual_heuristic_buds = find_virtual_buds_heuristic(cane_skeleton_np, interval_pixels=bud_interval_pixels)
        pruning_points = recommend_pruning_points_heuristic_cv(
            virtual_heuristic_buds, 
            cane_skeleton_np,
            pruning_offset_pixels=pruning_offset_pixels
        )
        
        skeleton_points_yx = np.column_stack(np.where(cane_skeleton_np > 0))
        skeleton_points_for_frontend = [{'x': int(p[1]), 'y': int(p[0])} for p in skeleton_points_yx]

        # --- 응답 데이터 구성 (NumPy 타입을 Python 기본 타입으로 확실히 변환) ---
        response_data = {
            "image_filename_processed": image_filename,
            "original_image_shape_hw": [int(s) for s in original_img_bgr.shape[:2]],
            "parameters_used": {
                "bud_interval_pixels": int(bud_interval_pixels),
                "pruning_offset_pixels": int(pruning_offset_pixels),
            },
            "virtual_buds": virtual_heuristic_buds, 
            "pruning_points": pruning_points,
            "cane_skeleton_points": skeleton_points_for_frontend,
        }
        
        return jsonify(response_data), 200

    except ValueError as ve: # 파라미터 변환 오류 등
        traceback.print_exc()
        return jsonify({"error": f"Invalid input or configuration: {str(ve)}"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    print(f"--- Flask Grape Pruning API (Heuristic) ---")
    print(f"Script location: {CURRENT_FLASK_SCRIPT_DIR}")
    print(f"Attempting to use Buds-Dataset base directory: {BASE_DATASET_DIR}")
    # ... (경로 존재 여부 확인 로그 추가 가능)
    print(f"Starting Flask server on http://0.0.0.0:8500") # 포트 변경 (FastAPI와 충돌 방지)
    app.run(host='0.0.0.0', port=8500, debug=True) # debug=True는 개발 중에만 사용