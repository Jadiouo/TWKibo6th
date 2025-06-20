package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import android.util.Log;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.util.HashSet;
import java.util.Arrays;
import java.util.Random;

// OpenCV imports
import org.opencv.aruco.Dictionary;
import org.opencv.aruco.Aruco;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.calib3d.Calib3d;
import org.opencv.imgproc.CLAHE;

public class YourService extends KiboRpcService {

    // 添加记录所有Landmark的变量
    private final Set<String> ALL_LANDMARKS = new HashSet<String>(Arrays.asList(
        "sushi", "applewatch", "picnicbasket", "guitar", "keyboard", "lunch", "piano", "toolbox"
    ));

    // 添加区域Landmark对应表
    private Map<Integer, Set<String>> areaLandmarksSet = new HashMap<>();

    // 用于随机选择的随机数生成器
    private Random random = new Random();

    // 增强日志记录的TAG前缀
    private final String TAG = "KiboRPC-" + this.getClass().getSimpleName();

    // Instance variables to store detection results across areas
    private Set<String> foundTreasures = new HashSet<>();
    private Set<String> foundLandmarks = new HashSet<>();  // Add this line
    private Map<String, Map<String, Integer>> areaLandmarks = new HashMap<>();

    // 添加这些新的成员变量来存储宝藏位置信息
    private Map<String, TreasureLocation> treasureLocations = new HashMap<>();
    private Map<Integer, TreasureLocation> areaLocations = new HashMap<>();

    // Area coordinates and orientations for all 4 areas 
    private final Point[] AREA_POINTS = {
            new Point(10.95d, -9.78d, 5.195d),         // Area 1
            new Point(10.925d, -8.3d, 4.94d),     // Area 2 
            new Point(10.925d, -7.925d, 4.56093d),     // Area 3
            new Point(10.666984d, -6.8525d, 4.945d)    // Area 4
    };

    private final Quaternion[] AREA_QUATERNIONS = {
            new Quaternion(0f, 0f, -0.707f, 0.707f), // Area 1
            new Quaternion(0f, 0.707f, 0f, 0.707f),  // Area 2
            new Quaternion(0f, 0.707f, 0f, 0.707f),  // Area 3
            new Quaternion(0f, 0f, 1f, 0f)           // Area 4
    };

    // 用于跟踪区域3是否已被处理
    private boolean area3Processed = false;

    // 定义宝藏位置类来存储坐标和朝向信息
    private class TreasureLocation {
        public Point position;
        public Quaternion orientation;
        
        public TreasureLocation(Point position, Quaternion orientation) {
            this.position = position;
            this.orientation = orientation;
        }
    }

    // 在类成员变量区域添加
    private Map<Integer, Set<String>> areaTreasure = new HashMap<>();

    @Override
    protected void runPlan1(){
        // 增加更详细的开始日志
        Log.i(TAG, "=== 任务开始 ===");
        Log.i(TAG, "当前时间: " + new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new java.util.Date()));

        // The mission starts.
        api.startMission();
        
        

        // 初始化区域Landmark和Treasure追踪
        for (int i = 1; i <= 4; i++) {
            areaTreasure.put(i, new HashSet<String>());
            areaLandmarksSet.put(i, new HashSet<String>());
        }
        
        // ========================================================================
        // CONFIGURABLE IMAGE PROCESSING PARAMETERS - EDIT HERE
        // ========================================================================

        Size cropWarpSize = new Size(640, 480);   // Size for cropped/warped image
        Size resizeSize = new Size(320, 320);     // Size for final processing

        // ========================================================================
        // PROCESS ALL 4 AREAS
        // ========================================================================

        // Loop through all 4 areas
        for (int areaIndex = 0; areaIndex < 4; areaIndex++) {
            int areaId = areaIndex + 1; // Area IDs are 1, 2, 3, 4

            // 其他区域的正常处理
            Point targetPoint = AREA_POINTS[areaIndex];
            Quaternion targetQuaternion = AREA_QUATERNIONS[areaIndex];

            Log.i(TAG, String.format("Moving to Area %d: Point(%.3f, %.3f, %.3f)",
                    areaId, targetPoint.getX(), targetPoint.getY(), targetPoint.getZ()));

            // 区域3检查 - 如果已经处理过就跳过
            if (areaId == 3 && area3Processed) {
                Log.i(TAG, "区域3已经在区域2处理时一起处理过，跳过");
                continue;
            }

            // 移动到指定位置
            // 原有的移动代码保持不变...
            switch (areaIndex) {
                case 0:
                    api.moveTo(new Point(10.925d, -9.85d, 4.695d), targetQuaternion, false); // Oasis1 cnter
                    api.moveTo(new Point(10.95d, -9.78d, 4.945d), targetQuaternion, false); // Scan point1
                    break;
                case 1:
                    api.moveTo(new Point(10.938d, -9.5d, 4.945d), targetQuaternion, false); // Oasis1&2 cnter
                    api.moveTo(new Point(11.175d, -8.975d, 5.195d), targetQuaternion, false); // Oasis2 cnter
                    api.moveTo(new Point(10.925d, -8.3d, 4.94d), targetQuaternion, false); // Scan point2
                    break;
                case 2:
                    api.moveTo(new Point(10.925d, -7.925d, 4.94d), targetQuaternion, false); // Scan point3
                    break;
                case 3:
                    api.moveTo(new Point(10.7d, -7.925d, 5.195d), targetQuaternion, false); // Oasis3 cnter
                    api.moveTo(new Point(10.925d, -7.4d, 4.945d), targetQuaternion, false); // Oasis3&4 cnter
                    api.moveTo(new Point(11.175d, -6.875d, 4.685d), targetQuaternion, false); // Oasis4 cnter
                    api.moveTo(new Point(10.925d, -6.8525d, 4.945d), targetQuaternion, false);// Scan point4
                    break;
                default:
                    break;
            }

            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                Log.w(TAG, "Sleep interrupted");
            }
            
            // 区域2特殊处理 - 同时处理区域2和区域3
            if (areaId == 2) {
                // 尝试同时处理区域2和区域3
                processDualAreas(cropWarpSize, resizeSize, areaTreasure);
                continue; // 跳过常规处理
            }

            Mat image = api.getMatNavCam();

            // 使用现有的imageEnhanceAndCrop处理图像
            Mat claHeBinImage = imageEnhanceAndCrop(image, cropWarpSize, resizeSize, areaId);

            // Initialize detection results for this area
            Map<String, Integer> landmark_items = new HashMap<>();
            Set<String> treasure_types = new HashSet<>();

            if (claHeBinImage != null) {
                Log.i(TAG, "Area " + areaId + ": Image enhancement and cropping successful");

                // Detect items using YOLO
                Object[] detected_items = detectitemfromcvimg(
                        claHeBinImage,
                        0.5f,      // conf_threshold
                        "lost",    // img_type ("lost" or "target")
                        0.45f,     // standard_nms_threshold
                        0.8f,      // overlap_nms_threshold
                        320        // img_size
                );

                // Extract results
                landmark_items = (Map<String, Integer>) detected_items[0];
                treasure_types = (Set<String>) detected_items[1];

                Log.i(TAG, "Area " + areaId + " - Landmark quantities: " + landmark_items);
                Log.i(TAG, "Area " + areaId + " - Treasure types: " + treasure_types);

                // Store results for later use
                areaLandmarks.put("area" + areaId, landmark_items);
                foundTreasures.addAll(treasure_types);

                // Add this line to store landmark types
                foundLandmarks.addAll(landmark_items.keySet());

                // Store treasure types for this area
                areaTreasure.get(areaId).addAll(treasure_types);

                // 记录区域位置信息
                areaLocations.put(areaId, new TreasureLocation(
                    AREA_POINTS[areaIndex],
                    AREA_QUATERNIONS[areaIndex]
                ));

                // 记录该区域发现的宝藏位置信息
                for (String treasureType : treasure_types) {
                    treasureLocations.put(treasureType, new TreasureLocation(
                        AREA_POINTS[areaIndex], 
                        AREA_QUATERNIONS[areaIndex]
                    ));
                }

                Log.i(TAG, "Area " + areaId + " treasure types: " + areaTreasure.get(areaId));

                // Clean up the processed image
                claHeBinImage.release();
            } else {
                Log.w(TAG, "Area " + areaId + ": Image enhancement failed - no markers detected or processing error");
            }

            // Clean up original image
            image.release();

            // ========================================================================
            // SET AREA INFO FOR THIS AREA
            // ========================================================================

            // Use the detected landmark items for area info
            String[] firstLandmark = getFirstLandmarkItem(landmark_items);
            if (firstLandmark != null) {
                String currentlandmark_items = firstLandmark[0];
                int landmarkCount = Integer.parseInt(firstLandmark[1]);

                // Set the area info with detected landmarks
                api.setAreaInfo(areaId, currentlandmark_items, landmarkCount);
                Log.i(TAG, String.format("Area %d: %s x %d", areaId, currentlandmark_items, landmarkCount));
            } else {
                Log.w(TAG, "Area " + areaId + ": No landmark items detected");
                // Set default if no detection
                api.setAreaInfo(areaId, "unknown", 0);
            }
        }

        // ========================================================================
        // LOG SUMMARY OF ALL AREAS
        // ========================================================================

        Log.i(TAG, "=== AREA PROCESSING SUMMARY ===");
        for (int i = 1; i <= 4; i++) {
            Log.i(TAG, "Area " + i + " treasures: " + areaTreasure.get(i));
            Log.i(TAG, "Area " + i + " landmarks: " + areaLandmarks.get("area" + i));
        }
        Log.i(TAG, "All found treasures: " + foundTreasures);
        Log.i(TAG, "All found landmarks: " + foundLandmarks);  // Add this line

        // ========================================================================
        // ASTRONAUT INTERACTION
        // ========================================================================

        // Move to the front of the astronaut and report rounding completion
        Point astronautPoint = new Point(11.143d, -6.7607d, 4.9654d);
        Quaternion astronautQuaternion = new Quaternion(0f, 0f, 0.707f, 0.707f);

        Log.i(TAG, "Moving to astronaut position");
        api.moveTo(astronautPoint, astronautQuaternion, false);
        api.reportRoundingCompletion();

        // Error handling verify markers are visible before proceeding
        boolean astronautMarkersOk = waitForMarkersDetection(2000, 200, "astronaut");

        if (astronautMarkersOk) {
            Log.i(TAG, "Astronaut markers confirmed - proceeding with target detection");
        } else {
            Log.w(TAG, "Astronaut markers not detected - proceeding anyway");
        }

        // ========================================================================
        // TARGET ITEM RECOGNITION
        // ========================================================================

        // Get target item image from astronaut
        Mat targetImage = api.getMatNavCam();

        // Process target image to identify what the astronaut is holding
        String targetTreasureType = processTargetImage(targetImage, resizeSize);
        Log.i(TAG, "识别结果 - 目标宝物类型: " + (targetTreasureType != null ? targetTreasureType : "null"));
        
        // 宝物识别失败的高级错误处理
        if (targetTreasureType == null || targetTreasureType.equals("unknown")) {
            Log.w(TAG, "无法识别目标宝物，尝试高级恢复策略");
            targetTreasureType = selectTargetBasedOnLandmarks();
            Log.i(TAG, "基于Landmark分析选择的备选宝物: " + targetTreasureType);
        }

        if (targetTreasureType != null && !targetTreasureType.equals("unknown")) {
            Log.i(TAG, "Target treasure identified: " + targetTreasureType);

            // Find which area contains this treasure
            int targetAreaId = findTreasureInArea(targetTreasureType, areaTreasure);

            if (targetAreaId > 0) {
                Log.i(TAG, "Target treasure '" + targetTreasureType + "' found in Area " + targetAreaId);

                // Notify recognition
                api.notifyRecognitionItem();

                // 使用精确导航功能移动到宝物前方0.9米处
                boolean positionSuccess = moveToTreasurePrecisely(targetTreasureType, targetAreaId);

                if (positionSuccess) {
                    // Take a snapshot of the target item
                    api.takeTargetItemSnapshot();
                    Log.i(TAG, "成功定位并拍摄目标宝物!");
                } else {
                    Log.w(TAG, "无法精确定位到目标宝物，使用常规拍摄");
                    api.takeTargetItemSnapshot();
                }

                Log.i(TAG, "Mission completed successfully!");
            } else {
                Log.w(TAG, "Target treasure '" + targetTreasureType + "' not found in any area");
                api.notifyRecognitionItem();
                api.takeTargetItemSnapshot();
            }
        } else {
            Log.w(TAG, "Could not identify target treasure from astronaut");
            api.notifyRecognitionItem();
            api.takeTargetItemSnapshot();
        }

        // Clean up target image
        targetImage.release();
    }

    /**
     * Process target image to identify the treasure type the astronaut is holding
     * @param targetImage Image from astronaut
     * @param resizeSize Processing size
     * @return Treasure type name or "unknown"
     */
    private String processTargetImage(Mat targetImage, Size resizeSize) {
        try {
            Log.i(TAG, "Processing target image from astronaut");

            // Save the target image for debugging
            api.saveMatImage(targetImage, "target_astronaut_raw.png");

            // Use the SAME processing pipeline as areas (ArUco detection + cropping + enhancement)
            Size cropWarpSize = new Size(640, 480);   // Same as area processing
            Mat processedTarget = imageEnhanceAndCrop(targetImage, cropWarpSize, resizeSize, 0); // Use 0 for target

            if (processedTarget != null) {
                Log.i(TAG, "Target image processing successful - markers detected and cropped");

                // Detect items using YOLO with "target" type - SAME as area processing
                Object[] detected_items = detectitemfromcvimg(
                        processedTarget,
                        0.3f,      // Lower confidence for target detection
                        "target",  // img_type for target
                        0.45f,     // standard_nms_threshold
                        0.8f,      // overlap_nms_threshold
                        320        // img_size
                );

                // Extract results - SAME as area processing
                Map<String, Integer> landmark_items = (Map<String, Integer>) detected_items[0];
                Set<String> treasure_types = (Set<String>) detected_items[1];

                Log.i(TAG, "Target - Landmark quantities: " + landmark_items);
                Log.i(TAG, "Target - Treasure types: " + treasure_types);

                if (!treasure_types.isEmpty()) {
                    String targetTreasure = treasure_types.iterator().next();
                    Log.i(TAG, "Target treasure detected: " + targetTreasure);
                    processedTarget.release();
                    return targetTreasure;
                }

                processedTarget.release();
            } else {
                Log.w(TAG, "Target image processing failed - no markers detected or processing error");
            }

            Log.w(TAG, "No treasure detected in target image");
            return "unknown";

        } catch (Exception e) {
            Log.e(TAG, "Error processing target image: " + e.getMessage());
            return "unknown";
        }
    }

    /**
     * Basic enhancement for target image (simpler than area processing)
     */
    private Mat enhanceTargetImage(Mat image, Size resizeSize) {
        try {
            // Resize to processing size
            Mat resized = new Mat();
            Imgproc.resize(image, resized, resizeSize);

            // Apply basic CLAHE enhancement
            Mat enhanced = new Mat();
            CLAHE clahe = Imgproc.createCLAHE();
            clahe.setClipLimit(2.0);
            clahe.setTilesGridSize(new Size(8, 8));
            clahe.apply(resized, enhanced);

            // Save enhanced target for debugging
            api.saveMatImage(enhanced, "target_astronaut_enhanced.png");

            resized.release();
            return enhanced;

        } catch (Exception e) {
            Log.e(TAG, "Error enhancing target image: " + e.getMessage());
            return null;
        }
    }

    /**
     * Find which area contains the specified treasure type
     * @param treasureType The treasure type to find
     * @param areaTreasure Map of area treasures
     * @return Area ID (1-4) or 0 if not found
     */
    private int findTreasureInArea(String treasureType, Map<Integer, Set<String>> areaTreasure) {
        for (int areaId = 1; areaId <= 4; areaId++) {
            Set<String> treasures = areaTreasure.get(areaId);
            if (treasures != null && treasures.contains(treasureType)) {
                return areaId;
            }
        }
        return 0; // Not found
    }

    /**
     * Method to detect items from CV image using YOLO - matches Python testcallyololib.py functionality
     * @param image Input OpenCV Mat image
     * @param conf Confidence threshold (e.g., 0.3f)
     * @param imgtype Image type: "lost" or "target"
     * @param standard_nms_threshold Standard NMS threshold (e.g., 0.45f)
     * @param overlap_nms_threshold Overlap NMS threshold for intelligent NMS (e.g., 0.8f)
     * @param img_size Image size for processing (e.g., 320)
     * @return Object array: [landmark_quantities (Map<String, Integer>), treasure_types (Set<String>)]
     */
    private Object[] detectitemfromcvimg(Mat image, float conf, String imgtype,
                                         float standard_nms_threshold, float overlap_nms_threshold, int img_size) {
        YOLODetectionService yoloService = null;
        try {
            Log.i(TAG, String.format("Starting YOLO detection - type: %s, conf: %.2f", imgtype, conf));

            // Initialize YOLO detection service
            yoloService = new YOLODetectionService(this);

            // Call detection with all parameters (matches Python simple_detection_example)
            YOLODetectionService.EnhancedDetectionResult result = yoloService.DetectfromcvImage(
                    image, imgtype, conf, standard_nms_threshold, overlap_nms_threshold
            );

            // Get Python-like result with class names
            Map<String, Object> pythonResult = result.getPythonLikeResult();

            // Extract landmark quantities (Map<String, Integer>) - matches Python detection['landmark_quantities']
            Map<String, Integer> landmarkQuantities = (Map<String, Integer>) pythonResult.get("landmark_quantities");
            if (landmarkQuantities == null) {
                landmarkQuantities = new HashMap<>();
            }

            // Extract treasure quantities and get the keys (types) - matches Python detection['treasure_quantities'].keys()
            Map<String, Integer> treasureQuantities = (Map<String, Integer>) pythonResult.get("treasure_quantities");
            if (treasureQuantities == null) {
                treasureQuantities = new HashMap<>();
            }
            Set<String> treasureTypes = new HashSet<>(treasureQuantities.keySet());

            // Log results (matches Python print statements)
            Log.i(TAG, "Landmark quantities: " + landmarkQuantities);
            Log.i(TAG, "Treasure types: " + treasureTypes);

            // Return as array: [landmark_quantities, treasure_types]
            // This matches Python: report_landmark.append(detection['landmark_quantities'])
            //                     store_treasure.append(detection['treasure_quantities'].keys())
            return new Object[]{landmarkQuantities, treasureTypes};

        } catch (Exception e) {
            Log.e(TAG, "Error in detectitemfromcvimg: " + e.getMessage(), e);
            // Return empty results on error
            return new Object[]{new HashMap<String, Integer>(), new HashSet<String>()};
        } finally {
            // Clean up YOLO service
            if (yoloService != null) {
                yoloService.close();
            }
        }
    }

    /**
     * Helper method to get the first landmark item and its count (matches Python usage pattern)
     * @param landmarkQuantities Map of landmark quantities
     * @return String array: [landmark_name, count_as_string] or null if empty
     */
    private String[] getFirstLandmarkItem(Map<String, Integer> landmarkQuantities) {
        if (landmarkQuantities != null && !landmarkQuantities.isEmpty()) {
            // Get first entry (matches Python landmark_items.keys()[0])
            Map.Entry<String, Integer> firstEntry = landmarkQuantities.entrySet().iterator().next();
            String landmarkName = firstEntry.getKey();
            Integer count = firstEntry.getValue();
            return new String[]{landmarkName, String.valueOf(count)};
        }
        return null;
    }

    /**
     * Enhanced image processing method that detects ArUco markers, crops region,
     * applies CLAHE enhancement, and binarizes the image
     * @param image Input image from NavCam
     * @param cropWarpSize Size for the cropped/warped image (e.g., 640x480)
     * @param resizeSize Size for the final processed image (e.g., 320x320)
     * @param areaId Area identifier for filename generation
     * @return Processed CLAHE + Otsu binarized image, or null if no markers detected
     */
    private Mat imageEnhanceAndCrop(Mat image, Size cropWarpSize, Size resizeSize, int areaId) {
        try {
            // 保存原始图像用于调试
            if (shouldSaveDebugImages(areaId)) {
                api.saveMatImage(image, "area_" + areaId + "_raw.png");
            }

            // 检测ArUco标记
            Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
            List<Mat> corners = new ArrayList<>();
            Mat ids = new Mat();
            Aruco.detectMarkers(image, dictionary, corners, ids);

            if (corners.isEmpty()) {
                Log.w(TAG, "未检测到ArUco标记");
                ids.release();
                return null;
            }

            // 保留最接近中心的标记
            Object[] filtered = keepClosestMarker(corners, ids, image);
            List<Mat> filteredCorners = (List<Mat>) filtered[0];
            Mat filteredIds = (Mat) filtered[1];
            
            // 清理原始资源
            for (Mat corner : corners) corner.release();
            ids.release();

            // 获取相机参数
            double[][] intrinsics = api.getNavCamIntrinsics();
            Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
            Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);
            cameraMatrix.put(0, 0, intrinsics[0]);
            distCoeffs.put(0, 0, intrinsics[1]);

            // 姿态估计
            Mat rvecs = new Mat();
            Mat tvecs = new Mat();
            float markerLength = 0.05f;
            Aruco.estimatePoseSingleMarkers(filteredCorners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

            if (rvecs.rows() > 0 && tvecs.rows() > 0) {
                Mat rvec = new Mat(3, 1, CvType.CV_64F);
                Mat tvec = new Mat(3, 1, CvType.CV_64F);
                rvecs.row(0).copyTo(rvec);
                tvecs.row(0).copyTo(tvec);

                // 处理裁剪区域
                Mat processedImage = processCropRegion(image, cameraMatrix, distCoeffs, 
                                                 rvec, tvec, cropWarpSize, resizeSize, areaId);
            
                // 清理资源
                rvec.release();
                tvec.release();
                cameraMatrix.release();
                distCoeffs.release();
                rvecs.release();
                tvecs.release();
                filteredIds.release();
                for (Mat corner : filteredCorners) corner.release();
            
                return processedImage;
            }

        // 清理资源
        cameraMatrix.release();
        distCoeffs.release();
        rvecs.release();
        tvecs.release();
        filteredIds.release();
        for (Mat corner : filteredCorners) corner.release();
        
        return null;
    } catch (Exception e) {
        Log.e(TAG, "图像处理错误: " + e.getMessage());
        return null;
    }
}

    /**
     * Helper method to process the crop region and apply CLAHE + binarization
     */
    private Mat processCropRegion(Mat image, Mat cameraMatrix, Mat distCoeffs, Mat rvec, Mat tvec, Size cropWarpSize, Size resizeSize, int areaId) {
        try {
            // Define crop area corners in 3D (manually adjusted)
            org.opencv.core.Point3[] cropCorners3D = {
                    new org.opencv.core.Point3(-0.0265, 0.0420, 0),    // Top-left
                    new org.opencv.core.Point3(-0.2385, 0.0420, 0),   // Top-right
                    new org.opencv.core.Point3(-0.2385, -0.1170, 0),  // Bottom-right
                    new org.opencv.core.Point3(-0.0265, -0.1170, 0)   // Bottom-left
            };

            MatOfPoint3f cropCornersMat = new MatOfPoint3f(cropCorners3D);
            MatOfPoint2f cropCorners2D = new MatOfPoint2f();

            // Convert distortion coefficients
            double[] distData = new double[5];
            distCoeffs.get(0, 0, distData);
            MatOfDouble distCoeffsDouble = new MatOfDouble();
            distCoeffsDouble.fromArray(distData);

            // Project crop corners to 2D
            Calib3d.projectPoints(cropCornersMat, rvec, tvec, cameraMatrix, distCoeffsDouble, cropCorners2D);
            org.opencv.core.Point[] cropPoints2D = cropCorners2D.toArray();

            if (cropPoints2D.length == 4) {
                // Create perspective transformation and get processed image with custom sizes
                Mat processedImage = cropEnhanceAndBinarize(image, cropPoints2D, cropWarpSize, resizeSize, areaId);

                // Clean up
                cropCornersMat.release();
                cropCorners2D.release();
                distCoeffsDouble.release();

                return processedImage;
            }

            // Clean up if crop failed
            cropCornersMat.release();
            cropCorners2D.release();
            distCoeffsDouble.release();

            return null;

        } catch (Exception e) {
            Log.e(TAG, "Error in processCropRegion: " + e.getMessage());
            return null;
        }
    }

    /**
 * Helper method to crop, enhance with CLAHE, and binarize the image
 * @param image Input image
 * @param cropPoints2D 2D points for perspective transformation
 * @param cropWarpSize Size for the cropped/warped image (configurable)
 * @param resizeSize Size for the final processed image (configurable)
 * @param areaId Area identifier for filename generation
 */
private Mat cropEnhanceAndBinarize(Mat image, org.opencv.core.Point[] cropPoints2D, 
                                  Size cropWarpSize, Size resizeSize, int areaId) {
    try {
        // 创建变换矩阵
        org.opencv.core.Point[] dstPoints = {
            new org.opencv.core.Point(0, 0),
            new org.opencv.core.Point(cropWarpSize.width - 1, 0),
            new org.opencv.core.Point(cropWarpSize.width - 1, cropWarpSize.height - 1),
            new org.opencv.core.Point(0, cropWarpSize.height - 1)
        };
        
        MatOfPoint2f srcPoints = new MatOfPoint2f(cropPoints2D);
        MatOfPoint2f dstPoints2f = new MatOfPoint2f(dstPoints);
        
        // 透视变换
        Mat perspectiveMatrix = Imgproc.getPerspectiveTransform(srcPoints, dstPoints2f);
        Mat croppedImage = new Mat();
        Imgproc.warpPerspective(image, croppedImage, perspectiveMatrix, cropWarpSize);
        
        // 调整大小
        Mat resizedImage = new Mat();
        Imgproc.resize(croppedImage, resizedImage, resizeSize);
        
        // CLAHE增强
        Mat claheImage = new Mat();
        CLAHE clahe = Imgproc.createCLAHE();
        clahe.setClipLimit(2.0);
        int gridSize = (int) Math.max(8, Math.min(resizeSize.width, resizeSize.height) / 40);
        clahe.setTilesGridSize(new Size(gridSize, gridSize));
        clahe.apply(resizedImage, claheImage);
        
        // 仅在调试模式保存增强后的图像
        if (shouldSaveDebugImages(areaId)) {
            api.saveMatImage(claheImage, String.format("area_%d_yolo_clahe.png", areaId));
        }
        
        // 清理资源
        srcPoints.release();
        dstPoints2f.release();
        perspectiveMatrix.release();
        croppedImage.release();
        resizedImage.release();
        
        return claheImage;
    } catch (Exception e) {
        Log.e(TAG, "图像裁剪增强错误: " + e.getMessage());
        return null;
    }
}

    /**
     * 保留最接近图像中心的标记
     */
    private Object[] keepClosestMarker(List<Mat> corners, Mat ids, Mat image) {
        if (corners.isEmpty()) {
            return new Object[]{new ArrayList<Mat>(), new Mat()};
        }
        
        // 单个标记情况
        if (corners.size() == 1) {
            List<Mat> clonedCorners = new ArrayList<>();
            clonedCorners.add(corners.get(0).clone());
            Mat clonedIds = new Mat();
            ids.copyTo(clonedIds);
            return new Object[]{clonedCorners, clonedIds};
        }
        
        // 计算图像中心
        double centerX = image.cols() / 2.0;
        double centerY = image.rows() / 2.0;
        
        // 找出最近的标记
        int closestIndex = 0;
        double minDistance = Double.MAX_VALUE;
        
        for (int i = 0; i < corners.size(); i++) {
            Mat corner = corners.get(i);
            float[] cornerData = new float[8];
            corner.get(0, 0, cornerData);
            
            // 计算标记中心
            double markerCenterX = 0, markerCenterY = 0;
            for (int j = 0; j < 4; j++) {
                markerCenterX += cornerData[j*2];
                markerCenterY += cornerData[j*2+1];
            }
            markerCenterX /= 4.0;
            markerCenterY /= 4.0;
            
            // 计算距离
            double distance = Math.sqrt(Math.pow(markerCenterX - centerX, 2) + 
                                  Math.pow(markerCenterY - centerY, 2));
            
            if (distance < minDistance) {
                minDistance = distance;
                closestIndex = i;
            }
        }
        
        // 创建结果
        List<Mat> filteredCorners = new ArrayList<>();
        filteredCorners.add(corners.get(closestIndex).clone());
        
        Mat filteredIds = new Mat(1, 1, CvType.CV_32S);
        int[] idValue = new int[1];
        ids.get(closestIndex, 0, idValue);
        filteredIds.put(0, 0, idValue);
        
        return new Object[]{filteredCorners, filteredIds};
    }

    /**
     * Verifies that ArUco markers are visible by taking pictures at regular intervals
     * @param maxWaitTimeMs Maximum time to wait (e.g., 2000)
     * @param intervalMs Interval between attempts (e.g., 200)
     * 
     * @return true if markers detected, false if timeout
     */
    private boolean waitForMarkersDetection(int maxWaitTimeMs, int intervalMs, String purpose) {
        int attempts = 0;
        int maxAttempts = maxWaitTimeMs / intervalMs;
        
        while (attempts < maxAttempts) {
            try {
                Mat image = api.getMatNavCam();
                if (image == null) continue;
                
                Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
                List<Mat> corners = new ArrayList<>();
                Mat ids = new Mat();
                
                Aruco.detectMarkers(image, dictionary, corners, ids);
                
                // 清理资源
                image.release();
                ids.release();
                
                // 检查结果
                if (!corners.isEmpty()) {
                    Log.i(TAG, purpose + ": 检测到" + corners.size() + "个标记");
                    for (Mat corner : corners) corner.release();
                    return true;
                }
                
                for (Mat corner : corners) corner.release();
                attempts++;
                Thread.sleep(intervalMs);
            } catch (Exception e) {
                Log.e(TAG, purpose + "标记检测出错: " + e.getMessage());
                attempts++;
                try { Thread.sleep(intervalMs); } catch (InterruptedException ie) { }
            }
        }
        
        Log.w(TAG, purpose + ": 超时未检测到标记");
        return false;
    }





    /**
     * 判断是否应该保存调试图像
     * @param areaId 区域ID
     * @return 是否保存
     */
    private boolean shouldSaveDebugImages(int areaId) {
        // 可以根据需要添加更复杂的逻辑，例如：
        // - 只对特定区域保存
        // - 根据系统时间决定
        // - 根据是否第一次处理决定
        
        // 默认不保存任何图像
        return false;
        
        // 如果需要特定的保存逻辑，可以修改如下：
        // return (areaId == 2 || areaId == 3); // 只保存区域2和3的图像
        // return api.getTimeRemaining() > 120; // 只在剩余时间充足时保存
    }

    // You can add your method.
    private String yourMethod(){
        return "your method";
    }



    /**
     * 同时处理区域2和区域3的图像
     * @param cropWarpSize 裁剪尺寸
     * @param resizeSize 调整尺寸
     * @param areaTreasure 区域宝藏映射
     */
    private void processDualAreas(Size cropWarpSize, Size resizeSize, Map<Integer, Set<String>> areaTreasure) {
        try {
            Log.i(TAG, "开始同时处理区域2和区域3");
            
            // 稳定后拍照
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            
            // 获取图像
            Mat dualImage = api.getMatNavCam();
            if (shouldSaveDebugImages(2)) {
                api.saveMatImage(dualImage, "areas_2_3_dual_view.png");
            }
            
            // 检测所有ArUco标记
            Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
            List<Mat> corners = new ArrayList<>();
            Mat ids = new Mat();
            
            Aruco.detectMarkers(dualImage, dictionary, corners, ids);
            
            if (corners.size() < 2) {
                Log.w(TAG, "未检测到足够的标记 (需要2个), 实际: " + corners.size());
                Log.w(TAG, "回退到单独处理区域2");
                
                // 清理资源
                dualImage.release();
                ids.release();
                for (Mat corner : corners) {
                    if (corner != null) corner.release();
                }
                
                // 处理单个区域2
                processSingleArea(2, cropWarpSize, resizeSize, areaTreasure);
                return;
            }
            
            Log.i(TAG, "检测到 " + corners.size() + " 个标记，计算位置");
            
            // 计算图像中心
            double centerX = dualImage.cols() / 2.0;
            double centerY = dualImage.rows() / 2.0;
            
            // 计算每个标记的中心点和到图像中心的距离
            List<MarkerInfo> markerInfoList = new ArrayList<>();
            
            for (int i = 0; i < corners.size(); i++) {
                Mat corner = corners.get(i);
                float[] cornerData = new float[8];  // 4个点，每个点2个坐标
                corner.get(0, 0, cornerData);
                
                // 计算标记中心
                double markerCenterX = 0;
                double markerCenterY = 0;
                for (int j = 0; j < 4; j++) {
                    markerCenterX += cornerData[j*2];
                    markerCenterY += cornerData[j*2+1];
                }
                markerCenterX /= 4.0;
                markerCenterY /= 4.0;
                
                // 计算到中心的距离
                double distance = Math.sqrt(Math.pow(markerCenterX - centerX, 2) + 
                                       Math.pow(markerCenterY - centerY, 2));
                
                int markerId = -1;
                if (!ids.empty() && i < ids.rows()) {
                    int[] idData = new int[1];
                    ids.get(i, 0, idData);
                    markerId = idData[0];
                }
                
                markerInfoList.add(new MarkerInfo(i, markerId, markerCenterX, markerCenterY, distance));
            }
            
            // 按距离排序
            java.util.Collections.sort(markerInfoList, new java.util.Comparator<MarkerInfo>() {
                @Override
                public int compare(MarkerInfo a, MarkerInfo b) {
                    return Double.compare(a.distance, b.distance);
                }
            });
            
            // 选择最接近中心的两个标记
            if (markerInfoList.size() >= 2) {
                // 第一个（距离最近）分配给区域2
                MarkerInfo area2Marker = markerInfoList.get(0);
                // 第二个（距离次近）分配给区域3
                MarkerInfo area3Marker = markerInfoList.get(1);
                
                // 处理相机参数
                double[][] intrinsics = api.getNavCamIntrinsics();
                Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
                Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);
                cameraMatrix.put(0, 0, intrinsics[0]);
                distCoeffs.put(0, 0, intrinsics[1]);
                
                // 估计标记的位姿
                Mat rvecs = new Mat();
                Mat tvecs = new Mat();
                float markerLength = 0.05f;  // 5cm标记尺寸
                
                Aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);
                
                Log.i(TAG, String.format("区域2标记索引: %d (ID:%d), 位置: (%.1f, %.1f), 距离中心: %.1f像素", 
                        area2Marker.index, area2Marker.id, area2Marker.centerX, area2Marker.centerY, area2Marker.distance));
                
                Log.i(TAG, String.format("区域3标记索引: %d (ID:%d), 位置: (%.1f, %.1f), 距离中心: %.1f像素", 
                        area3Marker.index, area3Marker.id, area3Marker.centerX, area3Marker.centerY, area3Marker.distance));
                
                // 处理区域2
                processAreaWithMarker(dualImage, cameraMatrix, distCoeffs, rvecs, tvecs, 
                                     area2Marker.index, cropWarpSize, resizeSize, 2, areaTreasure);
                
                // 处理区域3
                processAreaWithMarker(dualImage, cameraMatrix, distCoeffs, rvecs, tvecs, 
                                     area3Marker.index, cropWarpSize, resizeSize, 3, areaTreasure);
                
                // 标记区域3已处理
                area3Processed = true;
                
                // 清理资源
                cameraMatrix.release();
                distCoeffs.release();
                rvecs.release();
                tvecs.release();
            } else {
                Log.w(TAG, "处理后标记数量不足，回退到单独处理区域2");
                processSingleArea(2, cropWarpSize, resizeSize, areaTreasure);
            }
            
            // 清理资源
            dualImage.release();
            ids.release();
            for (Mat corner : corners) {
                if (corner != null) corner.release();
            }
            
        } catch (Exception e) {
            Log.e(TAG, "处理双区域时出错: " + e.getMessage(), e);
            // 回退到单独处理区域2
            processSingleArea(2, cropWarpSize, resizeSize, areaTreasure);
        }
    }

    /**
     * 存储ArUco标记信息的辅助类
     */
    private class MarkerInfo {
        int index;      // 在corners列表中的索引
        int id;         // 标记ID
        double centerX; // 标记中心X坐标
        double centerY; // 标记中心Y坐标
        double distance; // 到图像中心的距离
        
        public MarkerInfo(int index, int id, double centerX, double centerY, double distance) {
            this.index = index;
            this.id = id;
            this.centerX = centerX;
            this.centerY = centerY;
            this.distance = distance;
        }
    }

    /**
     * 用特定标记处理区域
     */
    private void processAreaWithMarker(Mat image, Mat cameraMatrix, Mat distCoeffs, 
                                      Mat rvecs, Mat tvecs, int markerIndex,
                                      Size cropWarpSize, Size resizeSize, 
                                      int areaId, Map<Integer, Set<String>> areaTreasure) {
        try {
            // 获取此标记的位姿
            Mat rvec = new Mat(3, 1, CvType.CV_64F);
            Mat tvec = new Mat(3, 1, CvType.CV_64F);
            
            rvecs.row(markerIndex).copyTo(rvec);
            tvecs.row(markerIndex).copyTo(tvec);
            
            // 处理裁剪区域并增强图像
            Mat processedImage = processCropRegion(image, cameraMatrix, distCoeffs, 
                                                 rvec, tvec, cropWarpSize, resizeSize, areaId);
            
            if (processedImage != null) {
                Log.i(TAG, "区域 " + areaId + " 图像处理成功");
                
                // 检测物品
                Object[] detected_items = detectitemfromcvimg(
                    processedImage,
                    0.3f,      // conf_threshold
                    "lost",    // img_type
                    0.45f,     // standard_nms_threshold
                    0.8f,      // overlap_nms_threshold
                    320        // img_size
                );
                
                // 提取结果
                Map<String, Integer> landmark_items = (Map<String, Integer>) detected_items[0];
                Set<String> treasure_types = (Set<String>) detected_items[1];
                
                Log.i(TAG, "区域 " + areaId + " - 地标数量: " + landmark_items);
                Log.i(TAG, "区域 " + areaId + " - 宝藏类型: " + treasure_types);
                
                // 存储结果
                areaLandmarks.put("area" + areaId, landmark_items);
                foundLandmarks.addAll(landmark_items.keySet());
                foundTreasures.addAll(treasure_types);
                areaTreasure.get(areaId).addAll(treasure_types);
                
                // 设置区域信息
                String[] firstLandmark = getFirstLandmarkItem(landmark_items);
                if (firstLandmark != null) {
                    String landmarkName = firstLandmark[0];
                    int landmarkCount = Integer.parseInt(firstLandmark[1]);
                    
                    api.setAreaInfo(areaId, landmarkName, landmarkCount);
                    Log.i(TAG, String.format("区域 %d: %s x %d", areaId, landmarkName, landmarkCount));
                } else {
                    api.setAreaInfo(areaId, "unknown", 0);
                    Log.w(TAG, "区域 " + areaId + ": 未检测到地标项目");
                }
                
                processedImage.release();
            } else {
                Log.w(TAG, "区域 " + areaId + " 图像处理失败");
                api.setAreaInfo(areaId, "unknown", 0);
            }
            
            // 清理资源
            rvec.release();
            tvec.release();
            
        } catch (Exception e) {
            Log.e(TAG, "处理区域 " + areaId + " 时出错: " + e.getMessage(), e);
            api.setAreaInfo(areaId, "unknown", 0);
        }
    }

    /**
     * 处理单个区域的辅助方法
     */
    private void processSingleArea(int areaId, Size cropWarpSize, Size resizeSize, 
                                  Map<Integer, Set<String>> areaTreasure) {
        // 移动到区域位置
        Point targetPoint = AREA_POINTS[areaId-1];
        Quaternion targetQuaternion = AREA_QUATERNIONS[areaId-1];
        
        Log.i(TAG, String.format("移动到区域 %d: (%.3f, %.3f, %.3f)",
                areaId, targetPoint.getX(), targetPoint.getY(), targetPoint.getZ()));
        
        api.moveTo(targetPoint, targetQuaternion, false);
        
        // 短暂等待稳定
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        
        // 获取图像并处理
        Mat image = api.getMatNavCam();
        if (image != null) {
            Mat processedImage = imageEnhanceAndCrop(image, cropWarpSize, resizeSize, areaId);
            
            // 处理检测结果
            if (processedImage != null) {
                // 检测物品
                Object[] detected_items = detectitemfromcvimg(
                        processedImage,
                        0.5f, "lost", 0.45f, 0.6f, 320);
                
                // 提取并保存结果
                Map<String, Integer> landmark_items = (Map<String, Integer>) detected_items[0];
                Set<String> treasure_types = (Set<String>) detected_items[1];
                
                areaLandmarks.put("area" + areaId, landmark_items);
                foundLandmarks.addAll(landmark_items.keySet());
                foundTreasures.addAll(treasure_types);
                areaTreasure.get(areaId).addAll(treasure_types);
                
                // 设置区域信息
                String[] firstLandmark = getFirstLandmarkItem(landmark_items);
                if (firstLandmark != null) {
                    api.setAreaInfo(areaId, firstLandmark[0], Integer.parseInt(firstLandmark[1]));
                    Log.i(TAG, String.format("区域 %d: %s x %d", 
                            areaId, firstLandmark[0], Integer.parseInt(firstLandmark[1])));
                } else {
                    api.setAreaInfo(areaId, "unknown", 0);
                }
                
                processedImage.release();
            } else {
                api.setAreaInfo(areaId, "unknown", 0);
            }
            
            image.release();
        }
    }

    /**
     * 精确移动到目标宝物前方，确保距离不超过0.9米且角度偏差不超过10度
     * @param treasureType 宝物类型
     * @param areaId 宝物所在区域
     * @return 是否成功定位
     */
    private boolean moveToTreasurePrecisely(String treasureType, int areaId) {
        Log.i(TAG, "开始精确移动到宝物 " + treasureType + " 前方");
        
        // 获取宝物位置信息
        TreasureLocation treasureLoc = treasureLocations.get(treasureType);
        
        // 如果没有特定宝物的位置信息，使用对应区域的位置
        if (treasureLoc == null) {
            Log.i(TAG, "未找到宝物" + treasureType + "的具体位置，使用区域" + areaId + "位置");
            treasureLoc = areaLocations.get(areaId);
            if (treasureLoc == null) {
                Log.e(TAG, "错误: 区域" + areaId + "位置信息不存在");
                if (areaId >= 1 && areaId <= 4) {
                    // 使用硬编码的备选位置
                    treasureLoc = new TreasureLocation(
                        AREA_POINTS[areaId-1], 
                        AREA_QUATERNIONS[areaId-1]
                    );
                    Log.w(TAG, "使用预定义区域坐标作为备选");
                } else {
                    Log.e(TAG, "无法确定目标位置，中止操作");
                    return false;
                }
            }
        }
        
        // 计算面对宝物的方向四元数（反方向）
        Quaternion faceOrientation = inverseFaceOrientation(treasureLoc.orientation);
        Log.i(TAG, "计算面对宝物的朝向");

        // 【修改】角度误差要求改为10度
        final double MAX_ANGLE_ERROR = 10.0;
        final double MAX_DISTANCE = 0.9;
        final double DISTANCE_TOLERANCE = 0.05;
        
        // 尝试三次定位
        for (int attempt = 1; attempt <= 3; attempt++) {
            Log.i(TAG, "尝试 " + attempt + "/3 定位到宝物前方");
            
            // 计算位于宝物前方的位置（距离稍微小于0.9米，留出余量）
            double targetDistance = 0.8; // 稍小于0.9米，以确保不会超出
            
            // 【修改】如果不是第一次尝试，基于当前位置计算调整后的位置
            if (attempt > 1) {
                // 获取当前位置和指向
                Point currentPosition = api.getRobotKinematics().getPosition();
                Quaternion currentOrientation = api.getRobotKinematics().getOrientation();
                
                // 计算当前距离和角度
                double currentDistance = getCurrentDistance(treasureLoc.position);
                double currentAngle = getCurrentAngle(faceOrientation);
                
                Log.i(TAG, String.format("调整前 - 距离: %.3fm, 角度偏差: %.1f°", 
                      currentDistance, currentAngle));
                
                // 根据当前情况调整目标距离
                if (currentDistance > MAX_DISTANCE) {
                    // 如果距离过大，需要靠近
                    targetDistance = 0.75;
                    Log.i(TAG, "距离过大，调整为更靠近宝物: " + targetDistance + "米");
                } else if (currentDistance < 0.6) {
                    // 如果距离过小，需要后退
                    targetDistance = 0.85;
                    Log.i(TAG, "距离过小，调整为远离宝物: " + targetDistance + "米");
                } else {
                    // 距离适中，微调
                    targetDistance = 0.8;
                }
                
                // 如果角度偏差大，先单独调整角度
                if (currentAngle > MAX_ANGLE_ERROR * 1.5) {
                    Log.i(TAG, "角度偏差较大，进行单独的角度调整");
                    
                    // 计算更精确的面向宝物的四元数
                    Quaternion refinedOrientation = calculateRefinedOrientation(
                        currentPosition, treasureLoc.position, currentOrientation);
                    
                    try {
                        // 原地调整角度
                        Log.i(TAG, "原地调整角度，面向宝物");
                        api.moveTo(currentPosition, refinedOrientation, false);
                        Log.i(TAG, "完成角度调整");
                        
                        // 等待稳定
                        Thread.sleep(500);
                        
                        // 检查角度调整效果
                        double newAngle = getCurrentAngle(faceOrientation);
                        Log.i(TAG, "角度调整后偏差: " + newAngle + "°");
                        
                        // 继续下一步位置调整
                    } catch (Exception e) {
                        Log.e(TAG, "角度调整失败: " + e.getMessage());
                    }
                }
            }
            
            // 计算或调整后的目标位置
            Point targetPosition = calculatePositionInFront(treasureLoc.position, treasureLoc.orientation, targetDistance);
            
            // 【修改】记录移动前状态
            Log.i(TAG, String.format("准备移动到目标位置(%.3f,%.3f,%.3f)，面对宝物",
                  targetPosition.getX(), targetPosition.getY(), targetPosition.getZ()));
            
            try {
                api.moveTo(targetPosition, faceOrientation, false);
                Log.i(TAG, "完成移动");
            } catch (Exception e) {
                Log.e(TAG, "移动过程中发生错误: " + e.getMessage());
                if (attempt == 3) {
                    // 最后一次尝试失败，使用无障碍模式
                    try {
                        Log.w(TAG, "尝试使用无障碍移动模式");
                        api.moveTo(targetPosition, faceOrientation, true);
                    } catch (Exception e2) {
                        Log.e(TAG, "无障碍移动也失败: " + e2.getMessage());
                        return false;
                    }
                }
                continue;
            }
            
            // 【修改】移动后检查距离和角度
            boolean distanceOK = checkDistance(treasureLoc.position, MAX_DISTANCE, DISTANCE_TOLERANCE);
            boolean angleOK = checkAngle(faceOrientation, MAX_ANGLE_ERROR);
            
            double actualDistance = getCurrentDistance(treasureLoc.position);
            double actualAngle = getCurrentAngle(faceOrientation);
            
            Log.i(TAG, String.format("定位检查结果 - 距离: %.3fm (要求≤%.1fm), 角度偏差: %.1f° (要求≤%.1f°)", 
                  actualDistance, MAX_DISTANCE, actualAngle, MAX_ANGLE_ERROR));
            
            // 如果距离和角度都符合要求，则成功
            if (distanceOK && angleOK) {
                Log.i(TAG, "成功定位到宝物前方! 距离和角度均符合要求");
                return true;
            }
            
            // 第三次尝试失败后，如果角度不符但距离符合，只调整角度
            if (attempt == 3 && distanceOK && !angleOK) {
                try {
                    Log.i(TAG, "最后尝试：保持位置，只调整角度");
                    Point currentPosition = api.getRobotKinematics().getPosition();
                    Quaternion refinedOrientation = calculateRefinedOrientation(
                        currentPosition, treasureLoc.position, api.getRobotKinematics().getOrientation());
                    
                    api.moveTo(currentPosition, refinedOrientation, false);
                    
                    // 再次检查角度
                    actualAngle = getCurrentAngle(faceOrientation);
                    Log.i(TAG, "最终角度调整后偏差: " + actualAngle + "°");
                    
                    // 如果角度仍不满足，但已是最后尝试，接受当前状态
                    return checkAngle(faceOrientation, MAX_ANGLE_ERROR);
                } catch (Exception e) {
                    Log.e(TAG, "最终角度调整失败: " + e.getMessage());
                }
            }
            
            // 短暂等待后进行下一次尝试
            if (attempt < 3) {
                try {
                    Log.i(TAG, "等待准备下一次调整...");
                    Thread.sleep(700);
                } catch (InterruptedException e) {
                    Log.w(TAG, "等待被中断");
                }
            }
        }
        
        // 三次尝试后仍未满足要求
        Log.w(TAG, "三次尝试后未能精确定位，将使用当前位置");
        return false;
    }

    /**
     * 基于Landmarks分析选择目标宝物
     * 当无法直接识别宇航员手中的宝物时使用
     * @return 选择的宝物类型
     */
    private String selectTargetBasedOnLandmarks() {
        Log.i(TAG, "基于Landmarks分析选择目标宝物");
        
        // 首先检查已确认的宝物区域
        List<Integer> areasWithTreasures = new ArrayList<>();
        for (int areaId = 1; areaId <= 4; areaId++) {
            Set<String> treasures = areaTreasure.get(areaId);
            if (treasures != null && !treasures.isEmpty()) {
                areasWithTreasures.add(areaId);
                Log.i(TAG, "区域" + areaId + "包含宝物: " + treasures);
            } else {
                Log.i(TAG, "区域" + areaId + "无宝物");
            }
        }
        
        // 发现的宇航员手持Landmark
        if (areaLandmarks.containsKey("target")) {
            Map<String, Integer> targetLandmarks = areaLandmarks.get("target");
            if (targetLandmarks != null && !targetLandmarks.isEmpty()) {
                Log.i(TAG, "宇航员手持Landmarks: " + targetLandmarks.keySet());
                
                // 分析哪些区域包含相同Landmark
                Map<Integer, Integer> areaMatchCount = new HashMap<>();
                for (String targetLandmark : targetLandmarks.keySet()) {
                    for (int areaId = 1; areaId <= 4; areaId++) {
                        Set<String> areaLandmarkSet = areaLandmarksSet.get(areaId);
                        if (areaLandmarkSet != null && areaLandmarkSet.contains(targetLandmark)) {
                            areaMatchCount.put(areaId, areaMatchCount.getOrDefault(areaId, 0) + 1);
                            Log.i(TAG, "Landmark " + targetLandmark + " 匹配区域 " + areaId);
                        }
                    }
                }
                
                // 找出匹配度最高的区域
                int bestAreaId = 0;
                int maxMatches = 0;
                for (Map.Entry<Integer, Integer> entry : areaMatchCount.entrySet()) {
                    if (entry.getValue() > maxMatches) {
                        maxMatches = entry.getValue();
                        bestAreaId = entry.getKey();
                    }
                }
                
                if (bestAreaId > 0 && areaTreasure.containsKey(bestAreaId) && 
                    !areaTreasure.get(bestAreaId).isEmpty()) {
                    // 从最佳匹配区域选择宝物
                    String selectedTreasure = areaTreasure.get(bestAreaId).iterator().next();
                    Log.i(TAG, "基于Landmark匹配选择区域" + bestAreaId + "宝物: " + selectedTreasure);
                    return selectedTreasure;
                }
            }
        }
        
        // 如果无法通过Landmark匹配，使用随机选择
        if (areasWithTreasures.isEmpty()) {
            Log.w(TAG, "所有区域均无检测到宝物，随机选择区域");
            int randomAreaId = random.nextInt(4) + 1;
            Log.i(TAG, "随机选择区域" + randomAreaId);
            return "random_area_" + randomAreaId;
        }
        
        // 从包含宝物的区域随机选择
        int selectedAreaIndex = random.nextInt(areasWithTreasures.size());
        int selectedAreaId = areasWithTreasures.get(selectedAreaIndex);
        Set<String> treasuresInArea = areaTreasure.get(selectedAreaId);
        
        // 从选定区域随机选择一个宝物
        String[] treasureArray = treasuresInArea.toArray(new String[0]);
        String selectedTreasure = treasureArray[random.nextInt(treasureArray.length)];
        
        Log.i(TAG, "从区域" + selectedAreaId + "随机选择宝物: " + selectedTreasure);
        return selectedTreasure;
    }

    /**
     * 计算面对目标的反向朝向四元数
     * @param targetOrientation 目标朝向四元数
     * @return 反向朝向四元数
     */
    private Quaternion inverseFaceOrientation(Quaternion targetOrientation) {
        // 提取方向向量
        double[] direction = quaternionToDirection(targetOrientation);
        
        // 反转方向向量
        direction[0] = -direction[0];
        direction[1] = -direction[1];
        direction[2] = -direction[2];
        
        Log.i(TAG, String.format("计算反向朝向 - 原始方向: [%.2f, %.2f, %.2f]", 
          -direction[0], -direction[1], -direction[2]));
        
        // 简单处理：调整原四元数以反向
        // 注意：对于简单的180度旋转，可以反转x、y、w分量，保留z分量
        return new Quaternion(
            -targetOrientation.getX(),
            -targetOrientation.getY(),
            targetOrientation.getZ(),
            -targetOrientation.getW()
        );
    }

    /**
     * 计算更精确的朝向四元数，使机器人正面朝向目标
     * @param currentPosition 当前位置
     * @param targetPosition 目标位置
     * @param currentOrientation 当前朝向
     * @return 调整后的四元数
     */
    private Quaternion calculateRefinedOrientation(Point currentPosition, Point targetPosition, Quaternion currentOrientation) {
        // 计算从当前位置到目标位置的向量
        double dx = targetPosition.getX() - currentPosition.getX();
        double dy = targetPosition.getY() - currentPosition.getY();
        double dz = targetPosition.getZ() - currentPosition.getZ();
        
        // 标准化向量
        double length = Math.sqrt(dx*dx + dy*dy + dz*dz);
        if (length > 0) {
            dx /= length;
            dy /= length;
            dz /= length;
        }
        
        Log.i(TAG, String.format("计算朝向向量: [%.3f, %.3f, %.3f]", -dx, -dy, -dz));
        
        // 从向量创建四元数（简化实现）
        // 反向指向（因为要面对目标）
        double[] v = {-dx, -dy, -dz};
        
        // 使用当前朝向作为基础，只进行微调
        double[] currentDir = quaternionToDirection(currentOrientation);
        double dotProduct = currentDir[0]*v[0] + currentDir[1]*v[1] + currentDir[2]*v[2];
        
        // 如果方向已经很接近，返回当前朝向
        if (dotProduct > 0.98) {
            return currentOrientation;
        }
        
        // 否则计算旋转轴和角度
        double[] crossProduct = {
            currentDir[1]*v[2] - currentDir[2]*v[1],
            currentDir[2]*v[0] - currentDir[0]*v[2],
            currentDir[0]*v[1] - currentDir[1]*v[0]
        };
        
        double crossLength = Math.sqrt(crossProduct[0]*crossProduct[0] + 
                                      crossProduct[1]*crossProduct[1] + 
                                      crossProduct[2]*crossProduct[2]);
        
        if (crossLength < 0.001) {
            return currentOrientation; // 向量共线，无法通过叉积确定旋转轴
        }
        
        // 标准化旋转轴
        crossProduct[0] /= crossLength;
        crossProduct[1] /= crossLength;
        crossProduct[2] /= crossLength;
        
        // 计算旋转角度（弧度）
        double angle = Math.acos(dotProduct);
        
        // 使用轴角表示法创建四元数
        double sinHalfAngle = Math.sin(angle / 2);
        double cosHalfAngle = Math.cos(angle / 2);
        
        return new Quaternion(
            (float)(crossProduct[0] * sinHalfAngle),
            (float)(crossProduct[1] * sinHalfAngle),
            (float)(crossProduct[2] * sinHalfAngle),
            (float)cosHalfAngle
        );
    }

    /**
     * 获取当前到目标位置的实际距离
     * @param targetPosition 目标位置
     * @return 实际距离(米)
     */
    private double getCurrentDistance(Point targetPosition) {
        Point currentPosition = api.getRobotKinematics().getPosition();
        double dx = currentPosition.getX() - targetPosition.getX();
        double dy = currentPosition.getY() - targetPosition.getY();
        double dz = currentPosition.getZ() - targetPosition.getZ();
        return Math.sqrt(dx*dx + dy*dy + dz*dz);
    }

    /**
     * 获取当前与目标朝向的角度差
     * 计算结果为面对面时的角度偏差
     * @param targetOrientation 目标朝向
     * @return 角度偏差(度)
     */
    private double getCurrentAngle(Quaternion targetOrientation) {
        Quaternion currentOrientation = api.getRobotKinematics().getOrientation();
        
        // 获取两个四元数表示的前向量
        double[] targetForward = quaternionToDirection(targetOrientation);
        double[] currentForward = quaternionToDirection(currentOrientation);
        
        // 计算两个向量间的夹角
        double dotProduct = -1 * (targetForward[0] * currentForward[0] + 
                             targetForward[1] * currentForward[1] + 
                             targetForward[2] * currentForward[2]);
    
        // 确保点积在有效范围内
        dotProduct = Math.min(Math.max(dotProduct, -1.0), 1.0);
    
        // 计算角度并转换为面对面偏差角度（0表示完全面对面，180表示背对背）
        double angleDegrees = Math.toDegrees(Math.acos(dotProduct));
        double deviationAngle = Math.abs(180 - angleDegrees);
    
        Log.i(TAG, String.format("角度计算 - 原始夹角: %.2f°, 面对面偏差: %.2f°", 
          angleDegrees, deviationAngle));
          
        return deviationAngle;
    }

    /**
     * 计算在指定点前方特定距离的位置
     * @param position 原始位置
     * @param orientation 朝向四元数
     * @param distance 前方距离（米）
     * @return 计算出的新位置
     */
    private Point calculatePositionInFront(Point position, Quaternion orientation, double distance) {
        // 提取四元数中的朝向信息
        double[] direction = quaternionToDirection(orientation);
        
        // 计算新位置
        double newX = position.getX() + direction[0] * distance;
        double newY = position.getY() + direction[1] * distance;
        double newZ = position.getZ() + direction[2] * distance;
        
        Log.d(TAG, String.format("从 (%.3f, %.3f, %.3f) 沿方向 [%.3f, %.3f, %.3f] 移动 %.1f米",
            position.getX(), position.getY(), position.getZ(), 
            direction[0], direction[1], direction[2], distance));
        
        return new Point(newX, newY, newZ);
    }

    /**
     * 检查与目标位置的距离是否符合要求（不超过最大距离）
     * @param targetPosition 目标位置
     * @param maxDistance 最大允许距离
     * @param tolerance 容许误差
     * @return 是否符合要求
     */
    private boolean checkDistance(Point targetPosition, double maxDistance, double tolerance) {
        double actualDistance = getCurrentDistance(targetPosition);
        boolean isWithinLimit = actualDistance <= maxDistance + tolerance;
        
        Log.i(TAG, String.format("距离检查 - 实际距离: %.3fm, 最大允许: %.3fm, 结果: %s",
          actualDistance, maxDistance + tolerance, isWithinLimit ? "符合" : "超限"));
          
        return isWithinLimit;
    }

    /**
     * 检查当前朝向与目标朝向的角度是否符合要求
     * @param targetOrientation 目标朝向
     * @param maxAngleDegrees 最大允许角度（度）
     * @return 是否符合要求
     */
    private boolean checkAngle(Quaternion targetOrientation, double maxAngleDegrees) {
        double actualAngle = getCurrentAngle(targetOrientation);
        boolean isWithinLimit = actualAngle <= maxAngleDegrees;
        
        Log.i(TAG, String.format("角度检查 - 偏差角度: %.2f°, 最大允许: %.2f°, 结果: %s", 
          actualAngle, maxAngleDegrees, isWithinLimit ? "符合" : "超限"));
          
        return isWithinLimit;
    }

    /**
     * 将四元数转换为前向量
     * @param q 四元数
     * @return 方向向量 [x, y, z]
     */
    private double[] quaternionToDirection(Quaternion q) {
        // 四元数分量
        double qx = q.getX();
        double qy = q.getY();
        double qz = q.getZ();
        double qw = q.getW();
        
        // 计算前方向向量（在Astrobee坐标系中）
        double[] direction = new double[3];
        
        // 四元数到前向量的转换
        direction[0] = 2 * (qx * qz + qw * qy);
        direction[1] = 2 * (qy * qz - qw * qx);
        direction[2] = 1 - 2 * (qx * qx + qy * qy);
        
        // 标准化向量
        double magnitude = Math.sqrt(direction[0] * direction[0] + 
                               direction[1] * direction[1] + 
                               direction[2] * direction[2]);
    
        if (magnitude > 0.0001) {
            direction[0] /= magnitude;
            direction[1] /= magnitude;
            direction[2] /= magnitude;
        }
        
        return direction;
    }
}