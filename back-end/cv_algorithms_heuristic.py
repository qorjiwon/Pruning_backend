import cv2
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any # Any 추가

# --- 색상 정의 (BGR 순서!) ---
# 중요: Buds-Dataset-main/SegmentationClassPNG 이미지에서 실제 색상값을 확인하고 정확하게 수정!
COLOR_BACKGROUND_BGR: List[int] = [0, 0, 0]
COLOR_CANE_BGR: List[int] = [0, 128, 0]      # RGB: (0, 128, 0) - 실제 값 확인 필요
COLOR_CORDON_BGR: List[int] = [0, 128, 128]  # RGB: (128, 128, 0) - 실제 값 확인 필요
COLOR_TRUNK_BGR: List[int] = [0, 0, 128]     # RGB: (128, 0, 0) - 실제 값 확인 필요

def extract_binary_mask_from_color(segmentation_mask_bgr: np.ndarray, 
                                   target_color_bgr: List[int], 
                                   tolerance: int = 15) -> np.ndarray:
    if not isinstance(target_color_bgr, (list, tuple)) or len(target_color_bgr) != 3:
        raise ValueError("target_color_bgr must be a list or tuple of 3 integers.")
    lower_bound = np.array([max(0, c - tolerance) for c in target_color_bgr], dtype=np.uint8)
    upper_bound = np.array([min(255, c + tolerance) for c in target_color_bgr], dtype=np.uint8)
    binary_mask = cv2.inRange(segmentation_mask_bgr, lower_bound, upper_bound)
    return binary_mask

def get_skeleton_cv(binary_mask: np.ndarray) -> np.ndarray:
    if not hasattr(cv2, 'ximgproc') or not hasattr(cv2.ximgproc, 'thinning'):
        print("Warning: cv2.ximgproc.thinning not available. Please install opencv-contrib-python.")
        print("Skeletonization will be skipped, returning original binary mask.")
        return binary_mask
    thinned = cv2.ximgproc.thinning(binary_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    return thinned

def find_virtual_buds_heuristic(cane_skeleton_img: np.ndarray, 
                               interval_pixels: int = 50,
                               max_buds: int = 20) -> List[Dict[str, int]]:
    virtual_buds: List[Dict[str, int]] = []
    points_yx = np.column_stack(np.where(cane_skeleton_img > 0)) 
    if len(points_yx) == 0: 
        return []
    sorted_points_yx = points_yx[np.lexsort((points_yx[:, 1], points_yx[:, 0]))] 
    last_added_point_xy: Optional[Tuple[int, int]] = None
    for y_coord, x_coord in sorted_points_yx:
        current_point_xy = (int(x_coord), int(y_coord))
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
        if len(virtual_buds) >= max_buds: 
            break
    return virtual_buds

def recommend_pruning_points_heuristic_cv(
    virtual_buds: List[Dict[str, Any]], 
    cane_skeleton_img: np.ndarray, 
    pruning_offset_pixels: int = 25,
    max_recommendations: int = 20,
    neighborhood_radius: int = 30
) -> List[Dict[str, Any]]: 
    candidate_pruning_points_with_priority: List[Dict[str, Any]] = [] 
    if not virtual_buds: 
        return []
    for i, bud_info in enumerate(virtual_buds):
        bud_x, bud_y = bud_info['x'], bud_info['y']
        candidate_pruning_y = bud_y - pruning_offset_pixels
        candidate_pruning_x = bud_x 
        point_data = {'x': bud_x, 'y': bud_y, 'is_offset_valid': False, 'original_bud_index': i}
        if 0 <= candidate_pruning_y < cane_skeleton_img.shape[0] and \
           0 <= candidate_pruning_x < cane_skeleton_img.shape[1] and \
           cane_skeleton_img[int(candidate_pruning_y), int(candidate_pruning_x)] > 0:
            point_data['x'] = int(candidate_pruning_x)
            point_data['y'] = int(candidate_pruning_y)
            point_data['is_offset_valid'] = True
        else:
            point_data['note'] = 'original_bud_location_offset_failed'
        priority_score = 0
        if point_data['is_offset_valid']:
            priority_score += 10
        priority_score += (cane_skeleton_img.shape[0] - bud_y) / float(cane_skeleton_img.shape[0] if cane_skeleton_img.shape[0] > 0 else 1) * 5 
        point_data['priority'] = priority_score
        candidate_pruning_points_with_priority.append(point_data)
    sorted_candidates = sorted(candidate_pruning_points_with_priority, key=lambda p: p.get('priority', 0), reverse=True)
    final_pruning_points: List[Dict[str, Any]] = []
    selected_positions_for_filter: List[Tuple[int, int]] = [] 
    for candidate in sorted_candidates:
        if len(final_pruning_points) >= max_recommendations:
            break
        candidate_pos = (candidate['x'], candidate['y'])
        too_close = False
        for selected_pos in selected_positions_for_filter:
            distance = math.sqrt((candidate_pos[0] - selected_pos[0])**2 + 
                                (candidate_pos[1] - selected_pos[1])**2)
            if distance < neighborhood_radius:
                too_close = True
                break
        if not too_close:
            point_to_add = {'x': candidate['x'], 'y': candidate['y']}
            if 'note' in candidate:
                point_to_add['note'] = candidate['note']
            final_pruning_points.append(point_to_add)
            selected_positions_for_filter.append(candidate_pos)
    return final_pruning_points

def limit_points(points: List[Dict[str, int]], max_points: int) -> List[Dict[str, int]]:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return [points[i] for i in indices]

def get_key_skeleton_points(cane_skeleton_img: np.ndarray, max_points: int = 10) -> List[Dict[str, int]]:
    key_points_list: List[Dict[str, int]] = []
    points_yx = np.column_stack(np.where(cane_skeleton_img > 0))
    if points_yx.size == 0:
        return key_points_list
    if len(points_yx) <= max_points:
        return [{'x': int(p[1]), 'y': int(p[0])} for p in points_yx]
    
    # 매우 단순화된 키 포인트 추출 (개선 필요)
    # 여기서는 단순히 골격 점들 중 일부를 균등하게 샘플링합니다.
    # 실제로는 끝점, 분기점 등을 찾는 알고리즘을 적용해야 합니다.
    simplified_points = limit_points([{'x': int(p[1]), 'y': int(p[0])} for p in points_yx], max_points)
    return simplified_points

# (선택) 디버깅용 시각화 함수 (cv_algorithms_heuristic.py 에 이미 있다면 그대로 사용)
def create_visualization_image(
    original_img_bgr: np.ndarray,
    cane_skeleton: np.ndarray,
    virtual_buds: List[Dict[str, int]],
    pruning_points: List[Dict[str, Any]], # Any for 'note'
    neighborhood_radius: int = 30, # 디버깅 시각화용
    output_path: str = "debug_heuristic_visualization.png"
) -> None:
    vis_img = original_img_bgr.copy()
    vis_img[cane_skeleton > 0] = [255, 0, 0]  # 골격: 파란색 (BGR)
    for bud in virtual_buds:
        cv2.circle(vis_img, (bud['x'], bud['y']), 5, (0, 255, 0), -1) # 눈: 초록색
    for point in pruning_points:
        cv2.drawMarker(vis_img, (point['x'], point['y']), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2) # 가지치기: 빨간색 X
        # # (선택) 주변 반경 시각화
        # overlay = vis_img.copy()
        # cv2.circle(overlay, (point['x'], point['y']), neighborhood_radius, (0, 100, 255), 1) 
        # cv2.addWeighted(overlay, 0.4, vis_img, 0.6, 0, vis_img)
    cv2.imwrite(output_path, vis_img)
    print(f"Heuristic visualization saved to {output_path}")