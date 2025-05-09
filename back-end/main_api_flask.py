import os
import traceback
from flask import Flask, request, jsonify
from typing import Optional
import cv2
import numpy as np
import math
import base64 # Base64 인코딩용
import io # 이미지 바이트 처리를 위해
from PIL import Image # Pillow 이미지 처리용

# OpenAI 라이브러리 임포트
import openai

# cv_algorithms_heuristic.py 파일에서 함수 및 변수 임포트
# Flask에서는 같은 디렉토리에 있다면 바로 임포트 가능
# cv_algorithms_heuristic.py 파일에서 함수 및 변수 임포트
from cv_algorithms_heuristic import (
    extract_binary_mask_from_color,
    get_skeleton_cv,
    find_virtual_buds_heuristic,
    recommend_pruning_points_heuristic_cv,
    get_key_skeleton_points, # 필요시 사용
    COLOR_CANE_BGR
)


app = Flask(__name__)

# --- OpenAI API 키 설정 (환경 변수에서 로드) ---
try:
    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    if not client.api_key:
        print("경고: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. ChatGPT 연동이 작동하지 않을 수 있습니다.")
except Exception as e:
    print(f"OpenAI 클라이언트 초기화 중 오류: {e}")
    client = None

# --- 경로 설정 (파일 구조에 맞게) ---
# 이 Flask 스크립트(main_api_flask.py)가 있는 디렉토리
CURRENT_FLASK_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Buds-Dataset-main 폴더는 이 스크립트의 부모 폴더(PURUNING) 아래에 위치
BASE_DATASET_DIR = os.path.join(CURRENT_FLASK_SCRIPT_DIR, '..', 'Buds-Dataset-main')
IMAGES_DIR = os.path.join(BASE_DATASET_DIR, 'Images')
MASKS_DIR = os.path.join(BASE_DATASET_DIR, 'SegmentationClassPNG')

# --- 이미지 Base64 인코딩 함수 ---
def encode_image_to_base64_str(image_path: str, image_format: str = "jpeg") -> Optional[str]:
    """
    이미지 파일을 읽어 Base64 문자열로 인코딩합니다.
    :param image_path: 이미지 파일 경로
    :param image_format: 이미지 포맷 (jpeg, png 등)
    :return: Base64 인코딩된 문자열 또는 오류 시 None
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/{image_format};base64,{encoded_string}"
    except Exception as e:
        print(f"이미지 Base64 인코딩 중 오류: {e}")
        return None

# --- ChatGPT 호출 함수 (이미지 입력 포함) ---
def get_chatgpt_comment_with_image(base64_image_str: Optional[str], num_pruning_points: int) -> Optional[str]:
    """
    ChatGPT를 호출하여 이미지와 가지치기 정보를 바탕으로 코멘트를 생성합니다.
    """
    if not client or not client.api_key:
        return "OpenAI API 키가 설정되지 않아 코멘트를 생성할 수 없습니다."
    if not base64_image_str:
        return "코멘트 생성을 위한 이미지가 제공되지 않았습니다."

    try:
        prompt_text = (
            f"이 포도나무 이미지에 대해 약 {num_pruning_points}개의 지점을 추천받아 가지치기를 수행할 예정입니다. "
            "이 이미지와 가지치기 정보를 바탕으로, 이 작업이 포도나무의 건강, 미래의 포도 수확량 및 품질 향상에 "
            "어떤 긍정적인 영향을 미칠 것으로 기대되는지, 농부에게 희망을 줄 수 있는 짧고 낙관적인 코멘트를 작성해주세요. "
            "(전문적인 수확량 예측이 아닌, 일반적인 기대 효과와 격려의 메시지)"
        )

        messages_payload = [
            {
                "role": "system",
                "content": "당신은 농업 기술과 식물 생장에 대해 긍정적이고 이해하기 쉬운 설명을 제공하는 AI 어시스턴트입니다."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": base64_image_str}}
                ]
            }
        ]
        
        # print(f"ChatGPT Request Payload: {messages_payload}") # 디버깅용

        chat_completion = client.chat.completions.create(
            messages=messages_payload,
            model="gpt-4o", # 또는 gpt-4-vision-preview 등 이미지 처리 가능 모델
            max_tokens=200, 
            temperature=0.7,
            n=1,
            stop=None,
        )
        response_text = chat_completion.choices[0].message.content.strip()
        return response_text
    except Exception as e:
        print(f"ChatGPT API 호출 중 오류 발생 (이미지 포함): {e}")
        return f"코멘트 생성 중 오류가 발생했습니다: {str(e)}"


@app.route('/get_recommendations_with_image_comment', methods=['GET'])
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

        # --- 원본 이미지를 Base64로 인코딩 ---
        # 이미지 포맷은 실제 파일에 맞게 (대부분 .png 또는 .jpeg)
        image_format = "png" if image_filename.lower().endswith(".png") else "jpeg"
        base64_image_data = encode_image_to_base64_str(original_image_path, image_format=image_format)

        # --- ChatGPT 호출 (Base64 이미지와 함께) ---
        chatgpt_comment = get_chatgpt_comment_with_image(base64_image_data, len(pruning_points))

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