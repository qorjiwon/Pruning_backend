# PURUNING/back-end/cv_algorithms_heuristic.py

import cv2
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any

# --- 색상 정의 (BGR 순서!) ---
COLOR_BACKGROUND_BGR: List[int] = [0, 0, 0]
COLOR_CANE_BGR: List[int] = [0, 128, 0]      # RGB: (0, 128, 0)
COLOR_CORDON_BGR: List[int] = [0, 128, 128]  # RGB: (128, 128, 0)
COLOR_TRUNK_BGR: List[int] = [0, 0, 128]     # RGB: (128, 0, 0)

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
        return binary_mask # 또는 np.zeros_like(binary_mask)
    thinned = cv2.ximgproc.thinning(binary_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    return thinned

def find_virtual_buds_heuristic(cane_skeleton_img: np.ndarray, 
                               interval_pixels: int = 50,
                               max_buds: int = 20) -> List[Dict[str, int]]:
    virtual_buds: List[Dict[str, int]] = []
    points_yx = np.column_stack(np.where(cane_skeleton_img > 0)) 
    if len(points_yx) == 0: 
        return []
    
    # y, x 순으로 정렬 (이 정렬이 항상 최선은 아님, 골격의 연결성을 따라가야 함)
    # 실제로는 골격의 한쪽 끝에서부터 추적하며 점을 찍어야 함
    sorted_points_yx = points_yx[np.lexsort((points_yx[:, 1], points_yx[:, 0]))] 

    last_added_point_xy: Optional[Tuple[int, int]] = None
    for y_coord, x_coord in sorted_points_yx:
        # NumPy 타입을 파이썬 기본 int로 명시적 변환
        current_x = int(x_coord)
        current_y = int(y_coord)
        current_point_xy = (current_x, current_y)
        
        if not virtual_buds: 
            virtual_buds.append({'x': current_x, 'y': current_y})
            last_added_point_xy = current_point_xy
        else:
            if last_added_point_xy: 
                # last_added_point_xy의 요소도 int여야 함
                prev_x, prev_y = int(last_added_point_xy[0]), int(last_added_point_xy[1])
                dist = math.sqrt(
                    (current_x - prev_x)**2 + 
                    (current_y - prev_y)**2
                )
                if dist >= interval_pixels:
                    virtual_buds.append({'x': current_x, 'y': current_y})
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
    # cordon_binary_mask: Optional[np.ndarray] = None, # 필요시 추가
    # original_image_shape_hw: Optional[Tuple[int, int]] = None # 필요시 추가
) -> List[Dict[str, Any]]: 
    candidate_pruning_points_with_priority: List[Dict[str, Any]] = [] 
    if not virtual_buds: 
        return []

    img_height, img_width = cane_skeleton_img.shape[:2] 
    # if original_image_shape_hw:
    #     img_height, img_width = original_image_shape_hw

    for i, bud_info in enumerate(virtual_buds):
        # bud_info의 x, y가 이미 int라고 가정 (find_virtual_buds_heuristic에서 변환)
        bud_x, bud_y = bud_info['x'], bud_info['y']
        
        candidate_pruning_y = bud_y - pruning_offset_pixels
        candidate_pruning_x = bud_x 
        # point_data의 x, y도 파이썬 기본 int로 저장되도록 함
        point_data = {'x': int(bud_x), 'y': int(bud_y), 'is_offset_valid': False, 'original_bud_index': int(i)}
        
        if 0 <= candidate_pruning_y < img_height and \
           0 <= candidate_pruning_x < img_width and \
           cane_skeleton_img[int(candidate_pruning_y), int(candidate_pruning_x)] > 0:
            point_data['x'] = int(candidate_pruning_x)
            point_data['y'] = int(candidate_pruning_y)
            point_data['is_offset_valid'] = True
        else:
            point_data['note'] = 'original_bud_location_offset_failed'
        
        priority_score = 0.0
        if point_data['is_offset_valid']:
            priority_score += 10.0
        
        image_center_y = img_height / 2.0 # float으로 연산
        # bud_y도 float으로 변환하여 연산 후 점수도 float
        center_proximity_score_y = (1.0 - abs(float(bud_y) - image_center_y) / (image_center_y if image_center_y > 0 else 1.0)) * 5.0
        priority_score += max(0.0, center_proximity_score_y)
        
        point_data['priority'] = float(priority_score) # float으로 저장
        candidate_pruning_points_with_priority.append(point_data)
            
    sorted_candidates = sorted(candidate_pruning_points_with_priority, key=lambda p: p.get('priority', 0.0), reverse=True)
    
    final_pruning_points: List[Dict[str, Any]] = []
    selected_positions_for_filter: List[Tuple[int, int]] = [] 
    
    for candidate in sorted_candidates:
        if len(final_pruning_points) >= max_recommendations:
            break
        # candidate의 x, y가 이미 int라고 가정
        candidate_x, candidate_y = candidate['x'], candidate['y']
        candidate_pos = (candidate_x, candidate_y)
        
        too_close = False
        for selected_pos in selected_positions_for_filter:
            # selected_pos의 요소도 int여야 함
            dist = math.sqrt((candidate_x - selected_pos[0])**2 + 
                                (candidate_y - selected_pos[1])**2)
            if dist < neighborhood_radius:
                too_close = True
                break
        if not too_close:
            point_to_add = {'x': candidate_x, 'y': candidate_y}
            if 'note' in candidate:
                point_to_add['note'] = str(candidate['note']) # note도 str로 확실히
            final_pruning_points.append(point_to_add)
            selected_positions_for_filter.append(candidate_pos)
            
    return final_pruning_points

def limit_points(points: List[Dict[str, int]], max_points: int) -> List[Dict[str, int]]:
    if len(points) <= max_points: return points
    # np.linspace 결과는 float이므로 int로 변환
    indices = np.linspace(0, len(points) - 1, max_points, dtype=int) 
    return [points[i] for i in indices]

def get_key_skeleton_points(cane_skeleton_img: np.ndarray, max_points: int = 10) -> List[Dict[str, int]]:
    points_yx = np.column_stack(np.where(cane_skeleton_img > 0))
    if points_yx.size == 0: return []
    
    # 모든 좌표를 파이썬 기본 int로 변환하여 리스트 생성
    all_points_dict_list = [{'x': int(p[1]), 'y': int(p[0])} for p in points_yx]

    if len(all_points_dict_list) <= max_points:
        return all_points_dict_list
    
    # limit_points 함수는 이미 딕셔너리 리스트를 받으므로 그대로 사용
    return limit_points(all_points_dict_list, max_points)


def draw_analysis_on_image(
    original_bgr_image: np.ndarray,
    virtual_buds: Optional[List[Dict[str, int]]] = None,
    pruning_points: Optional[List[Dict[str, Any]]] = None,
    key_skeleton_points: Optional[List[Dict[str, int]]] = None,
    cane_skeleton_mask: Optional[np.ndarray] = None,
    neighborhood_radius_for_pruning: Optional[int] = None
) -> np.ndarray:
    vis_image = original_bgr_image.copy()
    if cane_skeleton_mask is not None and np.any(cane_skeleton_mask):
        vis_image[cane_skeleton_mask > 0] = [255, 100, 100]
    if key_skeleton_points:
        for pt in key_skeleton_points: cv2.circle(vis_image, (int(pt['x']), int(pt['y'])), 4, (255, 200, 0), -1)
    if virtual_buds:
        for bud in virtual_buds: cv2.circle(vis_image, (int(bud['x']), int(bud['y'])), 6, (0, 200, 50), -1)
    if pruning_points:
        for point_info in pruning_points:
            pt_x, pt_y = int(point_info['x']), int(point_info['y']) # int 변환 확실히
            cv2.drawMarker(vis_image, (pt_x, pt_y), (0, 0, 220), cv2.MARKER_TILTED_CROSS, 15, 2)
            if neighborhood_radius_for_pruning:
                overlay = vis_image.copy()
                cv2.circle(overlay, (pt_x, pt_y), int(neighborhood_radius_for_pruning), (0, 0, 150, 50), -1)
                alpha = 0.1
                vis_image = cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0)
    return vis_image