import cv2
import numpy as np
import math
from typing import List, Dict, Tuple, Optional
import os

# --- 색상 정의 (BGR 순서!) ---
COLOR_BACKGROUND_BGR: List[int] = [0, 0, 0]
COLOR_CANE_BGR: List[int] = [0, 128, 0]      # RGB: (0, 128, 0)
COLOR_CORDON_BGR: List[int] = [0, 128, 128]  # RGB: (128, 128, 0)
COLOR_TRUNK_BGR: List[int] = [0, 0, 128]     # RGB: (128, 0, 0)

def extract_binary_mask_from_color(segmentation_mask_bgr: np.ndarray, target_color_bgr: List[int], tolerance: int = 15) -> np.ndarray:
    """
    세그멘테이션 마스크에서 특정 색상의 영역을 추출하여 이진 마스크로 반환합니다.
    
    Args:
        segmentation_mask_bgr: BGR 형식의 세그멘테이션 마스크 이미지
        target_color_bgr: 타겟 색상 (BGR 형식)
        tolerance: 색상 허용 오차
        
    Returns:
        이진 마스크 (값이 0 또는 255)
    """
    if not isinstance(target_color_bgr, (list, tuple)) or len(target_color_bgr) != 3:
        raise ValueError("target_color_bgr must be a list or tuple of 3 integers.")
    lower_bound = np.array([max(0, c - tolerance) for c in target_color_bgr], dtype=np.uint8)
    upper_bound = np.array([min(255, c + tolerance) for c in target_color_bgr], dtype=np.uint8)
    binary_mask = cv2.inRange(segmentation_mask_bgr, lower_bound, upper_bound)
    return binary_mask

def get_skeleton_cv(binary_mask: np.ndarray) -> np.ndarray:
    """
    이진 마스크의 골격을 추출합니다.
    
    Args:
        binary_mask: 이진 이미지 마스크
        
    Returns:
        골격화된 이진 이미지
    """
    if not hasattr(cv2, 'ximgproc') or not hasattr(cv2.ximgproc, 'thinning'):
        print("Warning: cv2.ximgproc.thinning not available. Please install opencv-contrib-python.")
        print("Skeletonization will be skipped, returning original binary mask.")
        return binary_mask
    thinned = cv2.ximgproc.thinning(binary_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    return thinned

def limit_points(points: List[Dict[str, int]], max_points: int) -> List[Dict[str, int]]:
    """
    포인트 목록을 최대 개수로 제한합니다.
    
    Args:
        points: 포인트 목록 ({x: int, y: int})
        max_points: 최대 포인트 수
        
    Returns:
        제한된 포인트 목록
    """
    if len(points) <= max_points:
        return points
    
    # 균등하게 샘플링
    indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return [points[i] for i in indices]

def find_virtual_buds_heuristic(cane_skeleton_img: np.ndarray, 
                               interval_pixels: int = 50,
                               max_buds: int = 20) -> List[Dict[str, int]]:
    """
    Cane 골격에서 가상 눈 위치를 찾습니다.
    
    Args:
        cane_skeleton_img: Cane 골격 이미지
        interval_pixels: 눈 사이의 간격 (픽셀)
        max_buds: 최대 가상 눈 개수
        
    Returns:
        가상 눈 위치 목록 ({x: int, y: int})
    """
    virtual_buds: List[Dict[str, int]] = []
    points_yx = np.column_stack(np.where(cane_skeleton_img > 0))
    if len(points_yx) == 0: 
        return []
    
    # y, x 순으로 정렬
    sorted_points_yx = points_yx[np.lexsort((points_yx[:, 1], points_yx[:, 0]))] 

    last_added_point_xy: Optional[Tuple[int, int]] = None
    current_segment_bud_count = 0

    for y, x in sorted_points_yx:
        current_point_xy = (int(x), int(y))
        
        if not virtual_buds:
            virtual_buds.append({'x': current_point_xy[0], 'y': current_point_xy[1]})
            last_added_point_xy = current_point_xy
            current_segment_bud_count += 1
        else:
            if last_added_point_xy:
                dist = math.sqrt(
                    (current_point_xy[0] - last_added_point_xy[0])**2 + 
                    (current_point_xy[1] - last_added_point_xy[1])**2
                )
                if dist >= interval_pixels:
                    virtual_buds.append({'x': current_point_xy[0], 'y': current_point_xy[1]})
                    last_added_point_xy = current_point_xy
                    current_segment_bud_count += 1
        
        # 최대 가상 눈 개수에 도달하면 중지
        if len(virtual_buds) >= max_buds:
            break
            
    return virtual_buds

def recommend_pruning_points_heuristic_cv(
    virtual_buds: List[Dict[str, any]],
    cane_skeleton_img: np.ndarray, 
    pruning_offset_pixels: int = 25,
    max_recommendations: int = 20,
    neighborhood_radius: int = 30
) -> List[Dict[str, int]]:
    """
    가상 눈을 기반으로 가지치기 추천 지점을 계산합니다.
    
    Args:
        virtual_buds: 가상 눈 위치 목록
        cane_skeleton_img: Cane 골격 이미지
        pruning_offset_pixels: 가지치기 오프셋 거리
        max_recommendations: 최대 추천 지점 수
        neighborhood_radius: 추천 지점 간 최소 거리
        
    Returns:
        가지치기 추천 지점 목록 ({x: int, y: int})
    """
    candidate_pruning_points: List[Dict[str, any]] = [] 
    if not virtual_buds: 
        return []

    for i, bud_info in enumerate(virtual_buds):
        bud_x, bud_y = bud_info['x'], bud_info['y']
        
        # 눈 위쪽에 오프셋 위치 시도
        candidate_pruning_y = bud_y - pruning_offset_pixels
        candidate_pruning_x = bud_x 

        point_data = {'x': bud_x, 'y': bud_y, 'is_offset_valid': False, 'original_bud_index': i}

        # 오프셋 위치가 골격 위에 있는지 확인
        if 0 <= candidate_pruning_y < cane_skeleton_img.shape[0] and \
           0 <= candidate_pruning_x < cane_skeleton_img.shape[1] and \
           cane_skeleton_img[int(candidate_pruning_y), int(candidate_pruning_x)] > 0:
            point_data['x'] = int(candidate_pruning_x)
            point_data['y'] = int(candidate_pruning_y)
            point_data['is_offset_valid'] = True
        else:
            point_data['note'] = 'original_bud_location_offset_failed'
        
        # 우선순위 점수 계산
        priority_score = 0
        
        # 유효한 오프셋에 더 높은 점수
        if point_data['is_offset_valid']:
            priority_score += 10
            
        # 이미지 상단에 위치한 눈(코돈에 더 가까운)에 더 높은 점수
        # 이미지 높이에 따라 스케일링하여 0-10점 사이 정규화
        height_factor = (cane_skeleton_img.shape[0] - bud_y) / cane_skeleton_img.shape[0]
        priority_score += height_factor * 10
        
        # 복잡도 점수 추가
        region_y_start = max(0, bud_y - 20)
        region_y_end = min(cane_skeleton_img.shape[0], bud_y + 20)
        region_x_start = max(0, bud_x - 20)
        region_x_end = min(cane_skeleton_img.shape[1], bud_x + 20)
        
        local_region = cane_skeleton_img[region_y_start:region_y_end, region_x_start:region_x_end]
        complexity_score = np.count_nonzero(local_region) / 400  # 정규화
        priority_score += complexity_score * 5  # 복잡도에 대해 최대 5점
        
        point_data['priority'] = priority_score
        candidate_pruning_points.append(point_data)
            
    # 우선순위별로 정렬(높은 순)
    sorted_candidates = sorted(candidate_pruning_points, key=lambda p: p.get('priority', 0), reverse=True)
    
    # 공간 필터링 적용 - 이미 선택된 지점과 충분히 떨어진 지점만 선택
    final_pruning_points: List[Dict[str, int]] = []
    selected_positions = []  # 이미 선택된 지점의 (x, y) 튜플 리스트
    
    for candidate in sorted_candidates:
        # 최대 추천 수에 도달했으면 중지
        if len(final_pruning_points) >= max_recommendations:
            break
            
        candidate_pos = (candidate['x'], candidate['y'])
        
        # 이 후보가 이미 선택된 지점에 너무 가까운지 확인
        too_close = False
        for selected_pos in selected_positions:
            distance = math.sqrt((candidate_pos[0] - selected_pos[0])**2 + 
                                (candidate_pos[1] - selected_pos[1])**2)
            if distance < neighborhood_radius:
                too_close = True
                break
                
        # 기존 지점과 너무 가깝지 않으면 선택에 추가
        if not too_close:
            point_data = {'x': candidate['x'], 'y': candidate['y']}
            if 'note' in candidate:
                point_data['note'] = candidate['note']
            
            final_pruning_points.append(point_data)
            selected_positions.append(candidate_pos)
            
    return final_pruning_points

def get_key_skeleton_points(cane_skeleton_img: np.ndarray, max_points: int = 10) -> List[Dict[str, int]]:
    """
    골격에서 중요한 포인트만 추출합니다(양끝점, 교차점 등).
    max_points를 초과하지 않도록 제한합니다.
    
    Args:
        cane_skeleton_img: Cane 골격 이미지
        max_points: 최대 포인트 수
        
    Returns:
        중요 골격 포인트 목록 ({x: int, y: int})
    """
    key_points = []
    if np.sum(cane_skeleton_img > 0) == 0:
        return key_points
        
    # 골격의 모든 점 좌표 가져오기
    points_yx = np.column_stack(np.where(cane_skeleton_img > 0))
    
    # 전체 포인트가 max_points보다 적으면 그대로 반환
    if len(points_yx) <= max_points:
        return [{'x': int(p[1]), 'y': int(p[0])} for p in points_yx]
    
    # 골격의 시작점과 끝점 탐색 (간단한 근사)
    # 실제로 더 복잡한 알고리즘 필요하지만 여기서는 단순화
    
    # y 좌표 기준 상단 3개, 하단 3개 점 선택
    sorted_by_y = sorted(points_yx, key=lambda p: p[0])
    top_points = sorted_by_y[:3]
    bottom_points = sorted_by_y[-3:]
    
    # x 좌표 기준 좌측 2개, 우측 2개 점 선택
    sorted_by_x = sorted(points_yx, key=lambda p: p[1])
    left_points = sorted_by_x[:2]
    right_points = sorted_by_x[-2:]
    
    # 교차점 찾기 (8-연결성 체크)
    junction_points = []
    kernel = np.ones((3, 3), np.uint8)
    for y, x in points_yx:
        if y > 0 and y < cane_skeleton_img.shape[0] - 1 and x > 0 and x < cane_skeleton_img.shape[1] - 1:
            # 3x3 이웃 추출
            neighbors = cane_skeleton_img[y-1:y+2, x-1:x+2].copy()
            neighbors[1, 1] = 0  # 중앙점 제거
            # 이웃이 3개 이상이면 교차점으로 간주
            if np.sum(neighbors > 0) >= 3:
                junction_points.append((y, x))
    
    # 중요 포인트들을 종합하고 중복 제거
    all_key_points = []
    for p in top_points + bottom_points + left_points + right_points + junction_points:
        if isinstance(p, tuple):
            point = {'x': int(p[1]), 'y': int(p[0])}
        else:
            point = {'x': int(p[1]), 'y': int(p[0])}
        
        # 중복 제거
        if point not in all_key_points:
            all_key_points.append(point)
    
    # 최대 포인트 수 제한
    return limit_points(all_key_points, max_points)

def create_visualization_image(
    original_img_bgr: np.ndarray,
    cane_skeleton: np.ndarray,
    virtual_buds: List[Dict[str, int]],
    pruning_points: List[Dict[str, int]],
    output_path: str = "debug_pruning_recommendation.png"
) -> None:
    """디버깅을 위한 가지치기 추천 시각화 생성"""
    vis_img = original_img_bgr.copy()
    
    # 골격을 파란색으로 그리기
    vis_img[cane_skeleton > 0] = [255, 0, 0]  # BGR에서 파란색
    
    # 가상 눈을 녹색 원으로 그리기
    for bud in virtual_buds:
        cv2.circle(vis_img, (bud['x'], bud['y']), 5, (0, 255, 0), -1)  # 녹색
    
    # 가지치기 지점을 빨간색 X 마커로 그리기
    for point in pruning_points:
        cv2.drawMarker(vis_img, (point['x'], point['y']), 
                      (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 2)  # 빨간색 X
        
        # 근접 반경을 반투명 원으로 그리기
        overlay = vis_img.copy()
        cv2.circle(overlay, (point['x'], point['y']), 
                  30, (0, 0, 255), 1)  # 근접 영역용 빨간색 원
        cv2.addWeighted(overlay, 0.4, vis_img, 0.6, 0, vis_img)
    
    cv2.imwrite(output_path, vis_img)
    print(f"시각화가 {output_path}에 저장되었습니다")

# --- 테스트용 코드 ---
if __name__ == '__main__':
    # 이 스크립트의 부모 폴더(back-end)의 부모 폴더(PURUNING) 아래 Buds-Dataset-main
    test_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Buds-Dataset-main')
    test_sample_image_name = 'ZED_image_left0.png' # 테스트할 이미지
    
    test_mask_path = os.path.join(test_base_dir, 'SegmentationClassPNG', test_sample_image_name)
    test_original_path = os.path.join(test_base_dir, 'Images', test_sample_image_name)

    if not os.path.exists(test_mask_path) or not os.path.exists(test_original_path):
        print(f"테스트 파일 경로 오류: {test_mask_path} 또는 {test_original_path}")
    else:
        print(f"--- cv_algorithms_heuristic.py 단위 테스트 시작 (이미지: {test_sample_image_name}) ---")
        seg_mask_bgr = cv2.imread(test_mask_path)
        original_img_bgr = cv2.imread(test_original_path)

        cane_mask = extract_binary_mask_from_color(seg_mask_bgr, COLOR_CANE_BGR)
        print(f"Cane 마스크 생성됨, 0이 아닌 픽셀 수: {np.count_nonzero(cane_mask)}")

        cane_skeleton = get_skeleton_cv(cane_mask)
        print(f"Cane 골격 생성됨, 0이 아닌 픽셀 수: {np.count_nonzero(cane_skeleton)}")

        # 가상 눈을 최대 20개로 제한
        v_buds = find_virtual_buds_heuristic(cane_skeleton, interval_pixels=30, max_buds=20)
        print(f"가상 눈 {len(v_buds)}개 생성")

        # 가지치기 추천 지점을 최대 15개로 제한
        p_points = recommend_pruning_points_heuristic_cv(
            v_buds, 
            cane_skeleton, 
            pruning_offset_pixels=15,
            max_recommendations=15,
            neighborhood_radius=30
        )
        print(f"추천 가지치기 지점 {len(p_points)}개 생성")

        # 골격 중요 포인트를 최대 5개로 제한
        key_skeleton_points = get_key_skeleton_points(cane_skeleton, max_points=5)
        print(f"골격 중요 포인트 {len(key_skeleton_points)}개 추출")

        # 시각화
        create_visualization_image(original_img_bgr, cane_skeleton, v_buds, p_points, "debug_cv_results.png")
        print("디버그 결과 이미지 'debug_cv_results.png' 저장됨.")
        
        # 총 포인트 수 확인
        total_points = len(p_points) + len(v_buds) + len(key_skeleton_points)
        print(f"총 포인트 수: {total_points}개 (최대 40개 제한)")
        print("--- cv_algorithms_heuristic.py 단위 테스트 완료 ---")