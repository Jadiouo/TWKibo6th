import torch
import cv2
import numpy as np
from ultralytics import YOLO
import torchvision
import os

def convert_detections_to_final_format(detections):
    """
    Convert detection candidates to final format
    Args:
        detections: List of detection candidates, each formatted as a dictionary with keys: 
        { 'class_id', 'bbox', 'confidence' }
        Each bbox is in [x_center, y_center, width, height] format.
    Returns:
        List of final detections formatted as dictionaries with keys:
        { 'class_id', 'x_center', 'y_center', 'width', 'height', 'confidence' }
        where x_center, y_center, width, height are floats.

    """
    final_detections = []
    for det in detections:
        final_detections.append({
            'class_id': det['class_id'],
            'x_center': det['bbox'][0].item(),
            'y_center': det['bbox'][1].item(),
            'width': det['bbox'][2].item(),
            'height': det['bbox'][3].item(),
            'confidence': det['confidence']
        })
    return final_detections

def apply_standard_nms(detections, nms_threshold):
    """
    Apply standard NMS to treasure detections (they shouldn't overlap much, to know class by position)
    Args:
        detections: 
        List of detection candidates, each formatted as a dictionary with keys: 
        { 'class_id', 'bbox', 'confidence' }

        num_detections: Also be called IOU threshold, which is the tolerance for overlapping boxes.
        But note that this is the standard case, so the nms_threshold will be the value like 
        your original setting on yolo_postprocess_pipeline() 
    """
    if len(detections) <= 1:
        return convert_detections_to_final_format(detections)
    
    # Convert to tensors for NMS
    boxes = []
    scores = []
    
    for det in detections:
        bbox = det['bbox']
        x_center, y_center, width, height = bbox
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        boxes.append([x1, y1, x2, y2])
        scores.append(det['confidence'])
    
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    
    # Apply standard NMS
    keep_indices = torchvision.ops.nms(boxes, scores, nms_threshold)
    
    # Return kept detections
    kept_detections = []
    for idx in keep_indices:
        kept_detections.append(detections[idx])
    
    return convert_detections_to_final_format(kept_detections)

def apply_landmark_intelligent_nms(detections, overlap_nms_threshold=0.6):
    """
    Apply intelligent NMS to landmark detections
    Simplified approach:
    1. Pick the class of the highest confidence detection
    2. Keep only detections of that class
    3. Apply standard NMS with overlap_nms_threshold to allow stacking within the class
    
    Args:
        detections: List of detection candidates with 'class_id', 'bbox', 'confidence'
        overlap_nms_threshold: IoU threshold for NMS within the selected class (higher = more stacking allowed)
    """
    # if only one detection, no need for NMS
    if len(detections) <= 1:
        return convert_detections_to_final_format(detections)
    
    print(f"        Applying intelligent NMS to {len(detections)} landmark detections")
    
    # Step 1: Find the highest confidence detection and its class
    highest_conf_detection = max(detections, key=lambda x: x['confidence'])
    selected_class = highest_conf_detection['class_id']
    
    print(f"        Selected class: {selected_class} (highest conf: {highest_conf_detection['confidence']:.3f})")
    
    # Step 2: Filter to only detections of the selected class
    same_class_detections = [det for det in detections if det['class_id'] == selected_class]
    
    print(f"        Detections of selected class: {len(same_class_detections)}/{len(detections)}")
    
    # Step 3: If only one detection of this class, return it
    if len(same_class_detections) <= 1:
        print(f"        Only one detection of selected class - no NMS needed")
        return convert_detections_to_final_format(same_class_detections)
    
    # Step 4: Apply standard NMS to same-class detections with overlap_nms_threshold
    boxes = []
    scores = []
    
    for det in same_class_detections:
        bbox = det['bbox']
        x_center, y_center, width, height = bbox
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        boxes.append([x1, y1, x2, y2])
        scores.append(det['confidence'])
    
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    
    # Apply NMS with overlap_nms_threshold to allow reasonable stacking
    keep_indices = torchvision.ops.nms(boxes, scores, overlap_nms_threshold)
    
    # Keep selected detections
    kept_detections = []
    for idx in keep_indices:
        kept_detections.append(same_class_detections[idx])
    
    print(f"        Landmarks kept after intelligent NMS: {len(kept_detections)}/{len(same_class_detections)} of class {selected_class}")
    
    return convert_detections_to_final_format(kept_detections)


def yolo_postprocess_pipeline(raw_tensor, conf_threshold=0.3, standard_nms_threshold=0.45, overlap_nms_threshold=0.8, img_size=320, imgtype="lost"):
    """
    Modified YOLO post-processing pipeline for treasure/landmark detection with intelligent NMS
    If input pixel size is 320x320, then the model output raw_tensor shape will be [1, 15, 2100]
    
    Args:
        raw_tensor: [1, 15, 2100] - Raw model output
        conf_threshold: Confidence threshold (0.3)
        nms_threshold: NMS IoU threshold (0.45) 
        img_size: Input image size (320)
        imgtype: "lost" or "target" - determines detection logic and NMS strategy
    
    Returns:
        Dictionary containing:
        - 'detections': List of final selected detections based on image type constraints
        - 'quantities': Dictionary of class_id -> detected count before selection
        - 'treasure_quantities': Dictionary of treasure class_id -> count
        - 'landmark_quantities': Dictionary of landmark class_id -> count
    """
    
    # All class Names
    all_class_names = ['coin', 'compass', 'coral', 'crystal', 'diamond', 'emerald', 
                    'fossil', 'key', 'letter', 'shell', 'treasure_box']
    # Define item categories
    treasures_names = ('crystal', 'diamond', 'emerald')
    landmark_names = ('coin', 'compass', 'coral', 'fossil', 'key', 'letter', 'shell', 'treasure_box')

    # Define treasure id and landmark id
    treasures_id = tuple([all_class_names.index(name) for name in treasures_names])
    landmark_id = tuple([all_class_names.index(name) for name in landmark_names])

    # print each torch layers min max
    for i in range(raw_tensor.shape[1]):
        layer = raw_tensor[ :, i, :] # [1, 2100] -> [2100]
        # print(f"Layer {i}: shape={layer.shape}, dtype={layer.dtype}")
        print(f"Layer {i}: min={layer.min().item():.6f}, max={layer.max().item():.6f}")

    total_features = 4 + len(all_class_names)  # 4 bbox + 11 class scores
    all_achors = (img_size // 32)**2 + (img_size // 16)**2 + (img_size // 8)**2  # 10x10 + 20x20 + 40x40 = 2100
    
    # Step 1: Convert tensor format
    if len(raw_tensor.shape) == 3 and raw_tensor.shape[1] == total_features:
        processed_tensor = raw_tensor.transpose(1, 2)  # [1, 2100, 15]
    else:
        processed_tensor = raw_tensor
    
    batch_size, num_detections, num_features = processed_tensor.shape
    print(f"Processing {num_detections} detection proposals")
    
    batch_detections = []
    
    for batch_idx in range(batch_size):
        detections = processed_tensor[batch_idx]  # [2100, 15]
        
        # Split into bbox coordinates and class scores
        bbox_coords = detections[:, :4]        # [2100, 4]
        class_scores = detections[:, 4:]       # [2100, 11]
        
        print(f"\nBatch {batch_idx}:")
        print(f"  Bbox coordinates shape: {bbox_coords.shape}")
        print(f"  Class scores shape: {class_scores.shape}")
        
        # Step 2: Apply confidence threshold to ALL classes
        conf_mask = class_scores > conf_threshold  # [2100, 11] boolean mask
        
        # Get detections where at least one class is above threshold
        any_class_above_threshold = torch.any(conf_mask, dim=1)  # [2100] boolean
        valid_detection_count = torch.sum(any_class_above_threshold).item()
        
        print(f"  Detections with any class above {conf_threshold}: {valid_detection_count}/{num_detections}")
        
        if valid_detection_count == 0:
            print("  No detections above confidence threshold!")
            continue
        
        # Filter detections and scores
        valid_bbox = bbox_coords[any_class_above_threshold]  # [N, 4]
        valid_scores = class_scores[any_class_above_threshold]  # [N, 11]
        valid_conf_mask = conf_mask[any_class_above_threshold]  # [N, 11]
        
        print(f"  Valid detections after filtering: {len(valid_bbox)}")
        
        # Step 3: For each valid detection, get class probabilities
        detection_candidates = []

        for det_idx in range(len(valid_bbox)):
            bbox = valid_bbox[det_idx] # [4] bbox coordinates
            scores = valid_scores[det_idx] # [11] class scores
            conf_mask_det = valid_conf_mask[det_idx] # [11] boolean mask
            
            # Get classes above threshold for this detection
            valid_classes = torch.where(conf_mask_det)[0] 
            valid_class_scores = scores[valid_classes]
            
            # Store possible detection classes for this bbox
            for class_idx, score in zip(valid_classes, valid_class_scores):
                detection_candidates.append({
                    'bbox': bbox,
                    'class_id': class_idx.item(),
                    'confidence': score.item(),
                    'detection_idx': det_idx
                })
        
        print(f"  Total detection candidates: {len(detection_candidates)}")

        # Step 4: Separate into treasure and landmark candidates
        treasure_detections = []
        landmark_detections = []
        
        for candidate in detection_candidates:
            class_id = candidate['class_id']
            if class_id in treasures_id:
                treasure_detections.append(candidate)
            elif class_id in landmark_id:
                landmark_detections.append(candidate)
        
        print(f"  Treasure detections: {len(treasure_detections)}")
        print(f"  Landmark detections: {len(landmark_detections)}")
        
        # ====================================================================
        # STEP 5: Apply image type constraints with intelligent NMS
        # ====================================================================
        
        final_detections = []
        treasure_quantities = {}
        landmark_quantities = {}
        all_quantities = {}
        
        if imgtype == "target":
            # Target item: exactly 2 landmark types + 1 treasure type
            # Apply STANDARD NMS to both treasures and landmarks
            print(f"\n  TARGET ITEM logic - applying STANDARD NMS:")
            
            # Apply standard NMS to treasures
            if treasure_detections:
                treasure_final_candidates = apply_standard_nms(treasure_detections, standard_nms_threshold)
                print(f"    Treasures after standard NMS: {len(treasure_final_candidates)}")
            else:
                treasure_final_candidates = []
            
            # Apply standard NMS to landmarks
            if landmark_detections:
                landmark_final_candidates = apply_standard_nms(landmark_detections, standard_nms_threshold)
                print(f"    Landmarks after standard NMS: {len(landmark_final_candidates)}")
            else:
                landmark_final_candidates = []
            
            # Count quantities after NMS
            for candidate in treasure_final_candidates:
                class_id = candidate['class_id']
                treasure_quantities[class_id] = treasure_quantities.get(class_id, 0) + 1
                all_quantities[class_id] = all_quantities.get(class_id, 0) + 1
            
            for candidate in landmark_final_candidates:
                class_id = candidate['class_id']
                landmark_quantities[class_id] = landmark_quantities.get(class_id, 0) + 1
                all_quantities[class_id] = all_quantities.get(class_id, 0) + 1
            
            # Sort by confidence for selection
            treasure_final_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            landmark_final_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Select exactly 1 treasure + 2 different landmark types
            if len(treasure_final_candidates) > 0 and len(landmark_final_candidates) >= 2:
                # Take highest confidence treasure (already in final format)
                selected_treasure = treasure_final_candidates[0]
                final_detections.append(selected_treasure)
                print(f"    Selected treasure: {all_class_names[selected_treasure['class_id']]} (conf: {selected_treasure['confidence']:.3f})")
                
                # Take top 2 landmark types (different classes, already in final format)
                selected_landmark_classes = set()
                for landmark in landmark_final_candidates:
                    if landmark['class_id'] not in selected_landmark_classes:
                        final_detections.append(landmark)
                        selected_landmark_classes.add(landmark['class_id'])
                        print(f"    Selected landmark: {all_class_names[landmark['class_id']]} (conf: {landmark['confidence']:.3f})")
                        
                        if len(selected_landmark_classes) == 2:
                            break
                
                if len(selected_landmark_classes) < 2:
                    print(f"    Warning: Only found {len(selected_landmark_classes)} landmark types, expected 2")
            else:
                print(f"    Error: Insufficient detections for target item")
                print(f"    Need: 1 treasure + 2 landmarks, Got: {len(treasure_final_candidates)} treasures + {len(landmark_final_candidates)} landmarks")
        
        elif imgtype == "lost":
            # Lost item: 1 landmark OR 1 landmark + 1 treasure
            print(f"\n  LOST ITEM logic - applying INTELLIGENT NMS:")
            
            if len(treasure_detections) > 0:
                # Case 1: 1 landmark + 1 treasure
                print(f"    Case 1: Treasure + Landmark detected")
                
                # Apply STANDARD NMS to treasures (remove overlapping of different classes)
                treasure_final_candidates = apply_standard_nms(treasure_detections, standard_nms_threshold)
                print(f"    Treasures after standard NMS: {len(treasure_final_candidates)}")
                
                # Apply INTELLIGENT NMS to landmarks (allow stacking of same class)
                if landmark_detections:
                    landmark_final_candidates = apply_landmark_intelligent_nms(
                        landmark_detections, 
                        # standard_nms_threshold= standard_nms_threshold,
                        overlap_nms_threshold=  overlap_nms_threshold # Higher threshold for allowing stacking
                    )
                    print(f"    Landmarks after intelligent NMS: {len(landmark_final_candidates)}")
                else:
                    landmark_final_candidates = []
                
                # Count quantities after NMS
                for candidate in treasure_final_candidates:
                    class_id = candidate['class_id']
                    treasure_quantities[class_id] = treasure_quantities.get(class_id, 0) + 1
                    all_quantities[class_id] = all_quantities.get(class_id, 0) + 1
                
                for candidate in landmark_final_candidates:
                    class_id = candidate['class_id']
                    landmark_quantities[class_id] = landmark_quantities.get(class_id, 0) + 1
                    all_quantities[class_id] = all_quantities.get(class_id, 0) + 1
                
                # Sort by confidence
                treasure_final_candidates.sort(key=lambda x: x['confidence'], reverse=True)
                landmark_final_candidates.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Select 1 treasure + 1 landmark (already in final format)
                if treasure_final_candidates:
                    selected_treasure = treasure_final_candidates[0]
                    final_detections.append(selected_treasure)
                    print(f"    Selected treasure: {all_class_names[selected_treasure['class_id']]} (conf: {selected_treasure['confidence']:.3f})")
                
                if landmark_final_candidates:
                    selected_landmark = landmark_final_candidates[0]
                    final_detections.append(selected_landmark)
                    print(f"    Selected landmark: {all_class_names[selected_landmark['class_id']]} (conf: {selected_landmark['confidence']:.3f})")
                else:
                    print(f"    Warning: Treasure found but no landmarks after intelligent NMS")
            
            else:
                # Case 2: Only landmarks
                print(f"    Case 2: Only landmarks detected")
                
                if landmark_detections:
                    # Apply INTELLIGENT NMS to landmarks (allow stacking of same class)
                    landmark_final_candidates = apply_landmark_intelligent_nms(
                        landmark_detections,
                        # standard_nms_threshold= standard_nms_threshold,
                        overlap_nms_threshold= overlap_nms_threshold  # Higher threshold for allowing stacking
                    )
                    print(f"    Landmarks after intelligent NMS: {len(landmark_final_candidates)}")
                    
                    # Count quantities after NMS
                    for candidate in landmark_final_candidates:
                        class_id = candidate['class_id']
                        landmark_quantities[class_id] = landmark_quantities.get(class_id, 0) + 1
                        all_quantities[class_id] = all_quantities.get(class_id, 0) + 1
                    
                    # Sort by confidence and select top landmark
                    landmark_final_candidates.sort(key=lambda x: x['confidence'], reverse=True)
                    
                    if landmark_final_candidates:
                        selected_landmark = landmark_final_candidates[0]
                        final_detections.append(selected_landmark)
                        print(f"    Selected landmark: {all_class_names[selected_landmark['class_id']]} (conf: {selected_landmark['confidence']:.3f})")
                    else:
                        print(f"    Error: No landmarks after intelligent NMS")
                else:
                    print(f"    Error: No landmark detections found")
        
        else:
            raise ValueError(f"Invalid imgtype: {imgtype}. Must be 'lost' or 'target'")
        
        # Log detected quantities
        print(f"\n  FINAL DETECTED QUANTITIES:")
        print(f"  Treasure quantities: {treasure_quantities}")
        for class_id, count in treasure_quantities.items():
            print(f"    {all_class_names[class_id]}: {count}")
        
        print(f"  Landmark quantities: {landmark_quantities}")
        for class_id, count in landmark_quantities.items():
            print(f"    {all_class_names[class_id]}: {count}")
        
        # ====================================================================
        # Step 6: Apply final NMS to selected detections (in case of overlaps)
        # ====================================================================
        
        if len(final_detections) > 1:
            print(f"\n  Applying final NMS to {len(final_detections)} selected detections")
            
            # Convert to tensors for NMS
            final_boxes = []
            final_scores = []
            
            for det in final_detections:
                # Extract coordinates from final format
                x_center, y_center, width, height = det['x_center'], det['y_center'], det['width'], det['height']
                # Convert center format to corner format
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                final_boxes.append([x1, y1, x2, y2])
                final_scores.append(det['confidence'])
            
            final_boxes = torch.tensor(final_boxes)
            final_scores = torch.tensor(final_scores)
            
            # Apply final NMS
            keep_indices = torchvision.ops.nms(final_boxes, final_scores, standard_nms_threshold)

            # Update final detections - already in correct format
            nms_detections = []
            for idx in keep_indices:
                nms_detections.append(final_detections[idx])
            
            print(f"    After final NMS: {len(nms_detections)} detections remain")
            
        else:
            # Already in correct format - no conversion needed
            nms_detections = final_detections
        
        # Create result dictionary
        result = {
            'detections': nms_detections,
            'quantities': all_quantities,
            'treasure_quantities': treasure_quantities,
            'landmark_quantities': landmark_quantities
        }
        batch_detections.append(result)
    
    # Return result dictionary or empty result
    if batch_detections:
        return batch_detections
    else:
        return {
            'detections': [],
            'quantities': {},
            'treasure_quantities': {},
            'landmark_quantities': {}
        }


def load_image_path(image_path, img_size=320):
    """
    Load and preprocess image from file path
    
    Args:
        image_path: Path to the image file
        img_size: Target size for resizing (default: 320)
    
    Returns:
        Preprocessed image as numpy array [H, W, C] in RGB format, normalized to [0,1]
    """
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (img_size, img_size))  # Resize to training size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    # Convert BGR to RGB
    image = image.astype(np.float32) / 255.0          # Normalize to [0,1]
    
    return image


def get_raw_yolo_tensor_flexible(model_path, cvimages):
    """
    Get raw YOLO tensor output - handles both single image and batch
    
    Args:
        model_path: Path to the YOLO model file
        cvimages: Single preprocessed CV image [H, W, C] OR list of images
    
    Returns:
        Raw model output tensor
    """
    # Load model
    model = YOLO(model_path)
    
    # Handle both single image and batch
    if isinstance(cvimages, list):
        if len(cvimages) > 1:
            # Batch processing
            batch_tensors = []
            for cvimage in cvimages:
                img_tensor = torch.from_numpy(cvimage).permute(2, 0, 1)
                batch_tensors.append(img_tensor)
            input_tensor = torch.stack(batch_tensors)  # [B, 3, H, W]
        elif len(cvimages) == 1:
            # Single image in list
            input_tensor = torch.from_numpy(cvimages[0]).permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError("cvimages list is empty")
    else:
        # Single image processing
        input_tensor = torch.from_numpy(cvimages).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    
    # Get raw output
    model.model.eval()
    with torch.no_grad():
        raw_output = model.model(input_tensor)
    
    # Return main output tensor
    if isinstance(raw_output, (list, tuple)):
        return raw_output[0]  # Main detection tensor
    else:
        return raw_output



def deal_with_result_detections(result_detections, class_names=None):
    """
    Deal with result detections and print them in a readable format
    Args:
        result_detections: List of detection results
        class_names: Optional list of class names for better readability
    Returns:
        detection_list:
            A list of dictionaries with class names as keys and int number of detections as values.
            if class_names is not provided, use the class_id as key
    """
    detection_list = []
    for result in result_detections:
        """
        Format of result:
        result = {
            'detections': nms_detections,
            'quantities': all_quantities,
            'treasure_quantities': treasure_quantities,
            'landmark_quantities': landmark_quantities
            }
        """

        treasure_quantities = result.get('treasure_quantities', {})
        landmark_quantities = result.get('landmark_quantities', {})
        treasure_result = {}
        landmark_result = {}
        print(treasure_quantities)
        print(landmark_quantities)
        # if 'treasure_quantities' is not {}

        if treasure_quantities:
            for key, value in treasure_quantities.items():
                class_name = class_names[key] if class_names else str(key)
                treasure_result[class_name] = value
        
        if landmark_quantities:
            for key, value in landmark_quantities.items():
                class_name = class_names[key] if class_names else str(key)
                landmark_result[class_name] = value
            

        result_dict = {
            'all_quantities': result.get('quantities', {}),
            'treasure_quantities': treasure_result, 
            'landmark_quantities': landmark_result
        }

        
        detection_list.append(result_dict)  

    return detection_list



def simple_detection_example(model_path, cv_img_list, 
                             img_type ="target", img_size=320, 
                             conf_threshold=0.3, standard_nms_threshold=0.45, 
                             overlap_nms_threshold=0.8):
    """
    Simple example for actual usage
    """
    print(f"\n{'='*60}")
    print("SIMPLE DETECTION EXAMPLE")
    print(f"{'='*60}")
    # image_path = r'E:\gitrepo\yolo-V8-main\kiborpc\0020.png'

    # stack cv_img with axis 0 of the raw tensor

    # Stack tensors along axis 0
    # raw_tensor = torch.stack(raw_tensor_list, dim=0)
    raw_tensor = get_raw_yolo_tensor_flexible(model_path, cv_img_list)

    # Get raw tensor
    # raw_tensor = get_raw_yolo_tensor(model_path, image_path)
    
    # Detect as lost item (recommended settings)
    result_detections = yolo_postprocess_pipeline(
        raw_tensor, 
        conf_threshold=conf_threshold, 
        standard_nms_threshold=standard_nms_threshold,
        overlap_nms_threshold=overlap_nms_threshold, 
        img_size=img_size,
        imgtype=img_type
    )

    class_names = ['coin', 'compass', 'coral', 'crystal', 'diamond', 'emerald', 
                    'fossil', 'key', 'letter', 'shell', 'treasure_box']
        # print(f"  {i+1}. {class_names[det['class_id']]} (confidence: {det['confidence']:.3f})")
    
    detection_list = deal_with_result_detections(result_detections, class_names)

    return detection_list


"""
if __name__ == "__main__":
    # Choose what to run:
    model_path = r'E:\gitrepo\yolo-V8-main\runs\detect\train10\weights\best.pt'
    image_base_folder = r'E:\gitrepo\yolo-V8-main\kiborpc'
    image_names = [ '0020.png', '0021.png']
    img_type = "target"  # or "target"
    img_size = 320
    conf_threshold = 0.3
    standard_nms_threshold = 0.45
    overlap_nms_threshold = 0.8
    
    print(f"\n{'='*60}")
    # Create cv_img_list from base folder and image names
    cv_img_list = []
    for image_name in image_names:
        image_path = os.path.join(image_base_folder, image_name)
        cv_img = load_image_path(image_path, img_size=img_size)
        cv_img_list.append(cv_img)

    # Option 1: Simple detection (recommended for actual use)
    detections = simple_detection_example(
        model_path=model_path, 
        cv_img_list=cv_img_list, 
        img_type=img_type,
        img_size=img_size, 
        conf_threshold=conf_threshold, 
        standard_nms_threshold=standard_nms_threshold, 
        overlap_nms_threshold=overlap_nms_threshold
    )
    print(f"\nFinal detection list: {detections}")
    
    # Option 2: Test different parameters (for experimentation)
    # test_pipeline_with_different_params()

"""