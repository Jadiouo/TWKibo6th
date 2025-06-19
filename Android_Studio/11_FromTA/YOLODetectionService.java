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

// OpenCV imports
import org.opencv.aruco.Dictionary;
import org.opencv.aruco.Aruco;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.calib3d.Calib3d;
import org.opencv.imgproc.CLAHE;

public class YourService extends KiboRpcService {

    private final String TAG = this.getClass().getSimpleName();

    // Instance variables to store detection results across areas
    private Set<String> foundTreasures = new HashSet<>();
    private Set<String> foundLandmarks = new HashSet<>();  // Add this line
    private Map<String, Map<String, Integer>> areaLandmarks = new HashMap<>();

    // Keep track of landmarks reported to the API
    private Set<String> reportedLandmarks = new HashSet<>();
    private final String[] ALL_POSSIBLE_LANDMARKS = {
            "coin", "compass", "coral", "crystal", "diamond", "emerald",
            "fossil", "key", "letter", "shell", "treasure_box"
    };


    // Area coordinates and orientations for all 4 areas 
    private final Point[] AREA_POINTS = {
            new Point(10.95d, -9.78d, 5.195d),         // Area 1
            new Point(10.925d, -8.875d, 4.56203d),     // Area 2 
            new Point(10.925d, -7.925d, 4.56093d),     // Area 3
            new Point(10.666984d, -6.8525d, 4.945d)    // Area 4
    };

    private final Quaternion[] AREA_QUATERNIONS = {
            new Quaternion(0f, 0f, -0.707f, 0.707f), // Area 1
            new Quaternion(0f, 0.707f, 0f, 0.707f),  // Area 2
            new Quaternion(0f, 0.707f, 0f, 0.707f),  // Area 3
            new Quaternion(0f, 0f, 1f, 0f)           // Area 4
    };

    @Override
    protected void runPlan1(){
        // Log the start of the mission.
        Log.i(TAG, "Start mission");

        // The mission starts.
        api.startMission();

        // Initialize area treasure tracking
        Map<Integer, Set<String>> areaTreasure = new HashMap<>();
        for (int i = 1; i <= 4; i++) {
            areaTreasure.put(i, new HashSet<String>());
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

            Log.i(TAG, "=== Processing Area " + areaId + " ===");

            // Move to the area
            Point targetPoint = AREA_POINTS[areaIndex];
            Quaternion targetQuaternion = AREA_QUATERNIONS[areaIndex];

            Log.i(TAG, String.format("Moving to Area %d: Point(%.3f, %.3f, %.3f)",
                    areaId, targetPoint.getX(), targetPoint.getY(), targetPoint.getZ()));

            api.moveTo(targetPoint, targetQuaternion, false);

            // Get a camera image
            Mat image = api.getMatNavCam();

            // Process the image for this area
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

                // Remember reported landmark and set area info
                reportedLandmarks.add(currentlandmark_items);
                api.setAreaInfo(areaId, currentlandmark_items, landmarkCount);
                Log.i(TAG, String.format("Area %d: %s x %d", areaId, currentlandmark_items, landmarkCount));
            } else {
                Log.w(TAG, "Area " + areaId + ": No landmark items detected, using fallback");
                String[] fallbackLandmark = getFallbackLandmark();
                api.setAreaInfo(areaId, fallbackLandmark[0], Integer.parseInt(fallbackLandmark[1]));
                Log.i(TAG, String.format("Area %d (FALLBACK): %s x %s", areaId, fallbackLandmark[0], fallbackLandmark[1]));
            }

            // Short delay between areas to ensure stability
            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                Log.w(TAG, "Sleep interrupted");
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

        if (targetTreasureType != null && !targetTreasureType.equals("unknown")) {
            Log.i(TAG, "Target treasure identified: " + targetTreasureType);

            // Find which area contains this treasure
            int targetAreaId = findTreasureInArea(targetTreasureType, areaTreasure);

            if (targetAreaId > 0) {
                Log.i(TAG, "Target treasure '" + targetTreasureType + "' found in Area " + targetAreaId);

                // Notify recognition
                api.notifyRecognitionItem();

                // Move back to the target area
                Point targetAreaPoint = AREA_POINTS[targetAreaId - 1];
                Quaternion targetAreaQuaternion = AREA_QUATERNIONS[targetAreaId - 1];

                Log.i(TAG, "Moving back to Area " + targetAreaId + " to get the treasure");
                api.moveTo(targetAreaPoint, targetAreaQuaternion, false);

                // Take a snapshot of the target item
                api.takeTargetItemSnapshot();

                Log.i(TAG, "Mission completed successfully!");
            } else {
                Log.w(TAG, "Target treasure '" + targetTreasureType + "' not found in any area");
                api.notifyRecognitionItem();
                int randomArea = getRandomAreaWithTreasure(areaTreasure);
                Point targetAreaPoint = AREA_POINTS[randomArea - 1];
                Quaternion targetAreaQuaternion = AREA_QUATERNIONS[randomArea - 1];
                Log.i(TAG, "Moving to random Area " + randomArea + " for snapshot");
                api.moveTo(targetAreaPoint, targetAreaQuaternion, false);
                api.takeTargetItemSnapshot();
            }
        } else {
            Log.w(TAG, "Could not identify target treasure from astronaut");
            api.notifyRecognitionItem();
            int randomArea = getRandomAreaWithTreasure(areaTreasure);
            Point randomPoint = AREA_POINTS[randomArea - 1];
            Quaternion randomQuat = AREA_QUATERNIONS[randomArea - 1];
            Log.i(TAG, "Moving to random Area " + randomArea + " for snapshot");
            api.moveTo(randomPoint, randomQuat, false);
            api.takeTargetItemSnapshot();
        }

        // Clean up target image
        targetImage.release();
    }

    @Override
    protected void runPlan2(){
        // write your plan 2 here.
    }

    @Override
    protected void runPlan3(){
        // write your plan 3 here.
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

            // Optional preprocessing for better accuracy
            Mat frame = EnhanceUtils.enhance(image);

            // Call detection with all parameters (matches Python simple_detection_example)
            YOLODetectionService.EnhancedDetectionResult result = yoloService.DetectfromcvImage(
                    frame, imgtype, conf, standard_nms_threshold, overlap_nms_threshold
            );
            frame.release();

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
            // Save original test image with area ID
            String rawImageFilename = "area_" + areaId + "_raw.png";
            api.saveMatImage(image, rawImageFilename);
            Log.i(TAG, "Raw image saved as " + rawImageFilename);

            // Initialize ArUco detection
            Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
            List<Mat> corners = new ArrayList<>();
            Mat ids = new Mat();

            // Detect markers
            Aruco.detectMarkers(image, dictionary, corners, ids);

            if (corners.size() > 0) {
                Log.i(TAG, "Detected " + corners.size() + " markers.");

                // Keep only the closest marker to image center
                Object[] filtered = keepClosestMarker(corners, ids, image);
                List<Mat> filteredCorners = (List<Mat>) filtered[0];
                Mat filteredIds = (Mat) filtered[1];

                // Clean up original corners and ids (now safe since we cloned the data)
                for (Mat corner : corners) {
                    corner.release();
                }
                ids.release();

                Log.i(TAG, "Using closest marker. Remaining markers: " + filteredCorners.size());

                // Get camera parameters
                double[][] intrinsics = api.getNavCamIntrinsics();
                Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
                Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);

                cameraMatrix.put(0, 0, intrinsics[0]);
                distCoeffs.put(0, 0, intrinsics[1]);
                distCoeffs.convertTo(distCoeffs, CvType.CV_64F);

                // Estimate pose for first marker
                Mat rvecs = new Mat();
                Mat tvecs = new Mat();
                float markerLength = 0.05f; // 5cm markers

                Aruco.estimatePoseSingleMarkers(filteredCorners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

                // Process first marker only
                Mat imageWithFrame = image.clone();
                Aruco.drawDetectedMarkers(imageWithFrame, filteredCorners, filteredIds);

                if (rvecs.rows() > 0 && tvecs.rows() > 0) {
                    Mat rvec = new Mat(3, 1, CvType.CV_64F);
                    Mat tvec = new Mat(3, 1, CvType.CV_64F);

                    rvecs.row(0).copyTo(rvec);
                    tvecs.row(0).copyTo(tvec);

                    // Convert to RGB and draw axis
                    Imgproc.cvtColor(imageWithFrame, imageWithFrame, Imgproc.COLOR_GRAY2RGB);
                    Aruco.drawAxis(imageWithFrame, cameraMatrix, distCoeffs, rvec, tvec, 0.1f);

                    // Save marker with frame using area ID
                    String markerFilename = "area_" + areaId + "_marker_0_with_frame.png";
                    api.saveMatImage(imageWithFrame, markerFilename);
                    Log.i(TAG, "Marker image saved as " + markerFilename);

                    // Process crop region and return enhanced image with custom sizes
                    Mat processedImage = processCropRegion(image, cameraMatrix, distCoeffs, rvec, tvec, cropWarpSize, resizeSize, areaId);

                    // Clean up
                    rvec.release();
                    tvec.release();
                    imageWithFrame.release();
                    cameraMatrix.release();
                    distCoeffs.release();
                    rvecs.release();
                    tvecs.release();

                    // Clean up filtered corners and ids
                    filteredIds.release();
                    for (Mat corner : filteredCorners) {
                        corner.release();
                    }

                    return processedImage;
                }

                // Clean up if pose estimation failed
                imageWithFrame.release();
                cameraMatrix.release();
                distCoeffs.release();
                rvecs.release();
                tvecs.release();
                filteredIds.release();
                for (Mat corner : filteredCorners) {
                    corner.release();
                }
            } else {
                Log.w(TAG, "No ArUco markers detected in image");
                // Clean up empty lists
                ids.release();
            }

            return null; // No markers detected

        } catch (Exception e) {
            Log.e(TAG, "Error in imageEnhanceAndCrop: " + e.getMessage());
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
    private Mat cropEnhanceAndBinarize(Mat image, org.opencv.core.Point[] cropPoints2D, Size cropWarpSize, Size resizeSize, int areaId) {
        try {
            // ========================================================================
            // STEP 1: Create cropped image with configurable size
            // ========================================================================

            // Define destination points for configurable rectangle
            org.opencv.core.Point[] dstPointsCrop = {
                    new org.opencv.core.Point(0, 0),                           // Top-left
                    new org.opencv.core.Point(cropWarpSize.width - 1, 0),      // Top-right
                    new org.opencv.core.Point(cropWarpSize.width - 1, cropWarpSize.height - 1),   // Bottom-right
                    new org.opencv.core.Point(0, cropWarpSize.height - 1)      // Bottom-left
            };

            // Create source and destination point matrices
            MatOfPoint2f srcPointsMat = new MatOfPoint2f(cropPoints2D);
            MatOfPoint2f dstPointsMatCrop = new MatOfPoint2f(dstPointsCrop);

            // Calculate perspective transformation matrix
            Mat perspectiveMatrixCrop = Imgproc.getPerspectiveTransform(srcPointsMat, dstPointsMatCrop);

            // Apply perspective transformation to get cropped image
            Mat croppedImage = new Mat();
            Imgproc.warpPerspective(image, croppedImage, perspectiveMatrixCrop, cropWarpSize);

            // Print min/max values of the cropped image
            Core.MinMaxLocResult minMaxResultCrop = Core.minMaxLoc(croppedImage);
            Log.i(TAG, String.format("Cropped image %.0fx%.0f - Min: %.2f, Max: %.2f",
                    cropWarpSize.width, cropWarpSize.height, minMaxResultCrop.minVal, minMaxResultCrop.maxVal));

            // Save the cropped image with area ID and dynamic filename
            String cropFilename = String.format("area_%d_cropped_region_%.0fx%.0f.png", areaId, cropWarpSize.width, cropWarpSize.height);
            api.saveMatImage(croppedImage, cropFilename);
            Log.i(TAG, "Cropped region saved as " + cropFilename);

            // ========================================================================
            // STEP 2: Resize to final processing size (configurable)
            // ========================================================================

            // Resize the cropped image to final size
            Mat resizedImage = new Mat();
            Imgproc.resize(croppedImage, resizedImage, resizeSize);

            // Save resized image with area ID
            String resizeFilename = String.format("area_%d_yolo_original_%.0fx%.0f.png", areaId, resizeSize.width, resizeSize.height);
            api.saveMatImage(resizedImage, resizeFilename);
            Log.i(TAG, "Resized image saved as " + resizeFilename);

            // ========================================================================
            // STEP 3: Apply CLAHE enhancement (FINAL OUTPUT)
            // ========================================================================

            // Apply CLAHE for better contrast enhancement
            Mat claheImage = new Mat();
            CLAHE clahe = Imgproc.createCLAHE();
            clahe.setClipLimit(2.0);  // Controls contrast enhancement

            // Adjust grid size based on image size
            int gridSize = (int) Math.max(8, Math.min(resizeSize.width, resizeSize.height) / 40);
            clahe.setTilesGridSize(new Size(gridSize, gridSize));

            clahe.apply(resizedImage, claheImage);

            // Print min/max values of the CLAHE-enhanced image
            Core.MinMaxLocResult claheMinMaxResult = Core.minMaxLoc(claheImage);
            Log.i(TAG, String.format("CLAHE enhanced image (%.0fx%.0f) - Min: %.2f, Max: %.2f",
                    resizeSize.width, resizeSize.height, claheMinMaxResult.minVal, claheMinMaxResult.maxVal));

            // Save CLAHE enhanced image with area ID
            String claheFilename = String.format("area_%d_yolo_clahe_%.0fx%.0f.png", areaId, resizeSize.width, resizeSize.height);
            api.saveMatImage(claheImage, claheFilename);
            Log.i(TAG, "CLAHE enhanced image saved as " + claheFilename);

            // ========================================================================
            // STEP 4: Apply Otsu's binarization (FOR DEBUG ONLY - NOT RETURNED)
            // ========================================================================

            // Apply Otsu's automatic threshold binarization for debugging purposes
            Mat binarizedOtsu = new Mat();
            double otsuThreshold = Imgproc.threshold(claheImage, binarizedOtsu, 0, 255,
                    Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);

            // Print min/max values and threshold of Otsu binarized image
            Core.MinMaxLocResult binaryOtsuResult = Core.minMaxLoc(binarizedOtsu);
            Log.i(TAG, String.format("Binary Otsu (%.1f) - Min: %.2f, Max: %.2f",
                    otsuThreshold, binaryOtsuResult.minVal, binaryOtsuResult.maxVal));

            // Save the Otsu binarized image for debugging purposes
            String binaryFilename = String.format("area_%d_debug_binary_otsu_%.0fx%.0f.png", areaId, resizeSize.width, resizeSize.height);
            api.saveMatImage(binarizedOtsu, binaryFilename);
            Log.i(TAG, String.format("Debug binary image saved as %s (threshold: %.1f)", binaryFilename, otsuThreshold));

            // ========================================================================
            // CLEANUP
            // ========================================================================

            // Clean up intermediate images (but NOT claheImage - that's our return value)
            srcPointsMat.release();
            dstPointsMatCrop.release();
            perspectiveMatrixCrop.release();
            croppedImage.release();
            resizedImage.release();
            binarizedOtsu.release();  // Release the debug binary image

            // Return the CLAHE enhanced image (instead of binary)
            return claheImage;

        } catch (Exception e) {
            Log.e(TAG, "Error in cropEnhanceAndBinarize: " + e.getMessage());
            return null;
        }
    }

    /**
     * FIXED: Keep only the marker closest to the image center
     * This version properly handles corner data format for ArUco
     * @param corners List of detected marker corners
     * @param ids Mat containing marker IDs
     * @param image Original image (to get center coordinates)
     * @return Object array: [filtered_corners, filtered_ids]
     */
    private Object[] keepClosestMarker(List<Mat> corners, Mat ids, Mat image) {
        if (corners.size() == 0) {
            return new Object[]{new ArrayList<Mat>(), new Mat()};
        }

        if (corners.size() == 1) {
            // For single marker, still clone the data to avoid memory issues
            List<Mat> clonedCorners = new ArrayList<>();
            clonedCorners.add(corners.get(0).clone());

            Mat clonedIds = new Mat();
            if (ids.rows() > 0) {
                ids.copyTo(clonedIds);
            }

            Log.i(TAG, "Single marker detected, using it.");
            return new Object[]{clonedCorners, clonedIds};
        }

        Log.i(TAG, "Multiple markers detected (" + corners.size() + "), finding closest to center...");

        // Calculate image center
        double imageCenterX = image.cols() / 2.0;
        double imageCenterY = image.rows() / 2.0;

        int closestIndex = 0;
        double minDistance = Double.MAX_VALUE;

        // Find the marker closest to image center
        for (int i = 0; i < corners.size(); i++) {
            Mat corner = corners.get(i);

            // Validate corner data format
            if (corner.rows() != 1 || corner.cols() != 4 || corner.channels() != 2) {
                Log.w(TAG, String.format("Invalid corner format for marker %d: %dx%d channels=%d",
                        i, corner.rows(), corner.cols(), corner.channels()));
                continue;
            }

            // Extract the 4 corner points safely
            float[] cornerData = new float[8]; // 4 points * 2 coordinates
            corner.get(0, 0, cornerData);

            // Calculate marker center (average of 4 corners)
            double markerCenterX = 0;
            double markerCenterY = 0;

            for (int j = 0; j < 4; j++) {
                markerCenterX += cornerData[j * 2];     // x coordinates
                markerCenterY += cornerData[j * 2 + 1]; // y coordinates
            }

            markerCenterX /= 4.0;
            markerCenterY /= 4.0;

            // Calculate distance to image center
            double distance = Math.sqrt(
                    Math.pow(markerCenterX - imageCenterX, 2) +
                            Math.pow(markerCenterY - imageCenterY, 2)
            );

            Log.i(TAG, String.format("Marker %d center: (%.1f, %.1f), distance: %.1f",
                    i, markerCenterX, markerCenterY, distance));

            if (distance < minDistance) {
                minDistance = distance;
                closestIndex = i;
            }
        }

        Log.i(TAG, "Closest marker: index " + closestIndex + ", distance: " + minDistance);

        // Create filtered results with properly cloned data
        List<Mat> filteredCorners = new ArrayList<>();
        Mat selectedCorner = corners.get(closestIndex);

        // Ensure the corner data is in the correct format and clone it
        if (selectedCorner.rows() == 1 && selectedCorner.cols() == 4 && selectedCorner.channels() == 2) {
            Mat clonedCorner = selectedCorner.clone();
            filteredCorners.add(clonedCorner);
        } else {
            Log.e(TAG, String.format("Selected corner has invalid format: %dx%d channels=%d",
                    selectedCorner.rows(), selectedCorner.cols(), selectedCorner.channels()));
            return new Object[]{new ArrayList<Mat>(), new Mat()};
        }

        // Also filter the IDs to match
        Mat filteredIds = new Mat();
        if (ids.rows() > closestIndex) {
            // Create a 1x1 matrix with the selected ID
            int[] idData = new int[1];
            ids.get(closestIndex, 0, idData);
            filteredIds = new Mat(1, 1, CvType.CV_32S);
            filteredIds.put(0, 0, idData);
        }

        return new Object[]{filteredCorners, filteredIds};
    }

    /**
     * Verifies that ArUco markers are visible by taking pictures at regular intervals
     * @param maxWaitTimeMs Maximum time to wait (e.g., 2000)
     * @param intervalMs Interval between attempts (e.g., 200)
     * @param debugPrefix Prefix for saved debug images (e.g., "astronaut")
     * @return true if markers detected, false if timeout
     */
    private boolean waitForMarkersDetection(int maxWaitTimeMs, int intervalMs, String debugPrefix) {
        boolean markersDetected = false;
        int maxAttempts = maxWaitTimeMs / intervalMs;
        int attempts = 0;
        long startTime = System.currentTimeMillis();

        Log.i(TAG, String.format("Starting marker detection verification - max %dms, interval %dms",
                maxWaitTimeMs, intervalMs));

        while (!markersDetected && attempts < maxAttempts) {
            try {
                // Take a picture
                Mat testImage = api.getMatNavCam();

                if (testImage != null) {
                    // Initialize ArUco detection
                    Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
                    List<Mat> corners = new ArrayList<>();
                    Mat ids = new Mat();

                    // Detect markers
                    Aruco.detectMarkers(testImage, dictionary, corners, ids);

                    if (corners.size() > 0) {
                        markersDetected = true;
                        long elapsedTime = System.currentTimeMillis() - startTime;
                        Log.i(TAG, String.format("SUCCESS: %d markers detected after %d attempts (%.1fs)",
                                corners.size(), attempts + 1, elapsedTime / 1000.0));

                        // Save successful image for debugging
                        api.saveMatImage(testImage, debugPrefix + "_markers_detected.png");
                    } else {
                        Log.d(TAG, String.format("Attempt %d/%d: No markers detected", attempts + 1, maxAttempts));
                    }

                    // Clean up ArUco detection resources
                    for (Mat corner : corners) {
                        corner.release();
                    }
                    ids.release();

                    // Clean up test image
                    testImage.release();
                } else {
                    Log.w(TAG, "Failed to get image from camera on attempt " + (attempts + 1));
                }

                attempts++;

                // Wait before next attempt (only if not the last attempt)
                if (!markersDetected && attempts < maxAttempts) {
                    Thread.sleep(intervalMs);
                }

            } catch (InterruptedException e) {
                Log.w(TAG, "Sleep interrupted during marker detection");
                break;
            } catch (Exception e) {
                Log.e(TAG, "Error during marker detection attempt " + (attempts + 1) + ": " + e.getMessage());
                attempts++;

                // Still wait before next attempt
                if (attempts < maxAttempts) {
                    try {
                        Thread.sleep(intervalMs);
                    } catch (InterruptedException ie) {
                        Log.w(TAG, "Sleep interrupted after error");
                        break;
                    }
                }
            }
        }

        // Log final result
        long totalTime = System.currentTimeMillis() - startTime;
        if (markersDetected) {
            Log.i(TAG, String.format("%s position verified - markers visible", debugPrefix));
            return true;
        } else {
            Log.w(TAG, String.format("WARNING: No markers detected at %s after %d attempts (%.1fs)",
                    debugPrefix, attempts, totalTime / 1000.0));
            return false;
        }
    }

    // Choose a landmark that hasn't been reported yet
    private String[] getFallbackLandmark() {
        List<String> unreported = new ArrayList<>();
        for (String lm : ALL_POSSIBLE_LANDMARKS) {
            if (!reportedLandmarks.contains(lm)) {
                unreported.add(lm);
            }
        }

        if (unreported.isEmpty()) {
            unreported = Arrays.asList(ALL_POSSIBLE_LANDMARKS);
        }

        int idx = (int) (System.currentTimeMillis() % unreported.size());
        String landmark = unreported.get(idx);
        reportedLandmarks.add(landmark);
        return new String[]{landmark, "1"};
    }

    // Pick a random area that contains treasures
    private int getRandomAreaWithTreasure(Map<Integer, Set<String>> areaTreasure) {
        List<Integer> areas = new ArrayList<>();
        for (int i = 1; i <= 4; i++) {
            if (areaTreasure.get(i) != null && !areaTreasure.get(i).isEmpty()) {
                areas.add(i);
            }
        }

        if (areas.isEmpty()) {
            return (int) (System.currentTimeMillis() % 4) + 1;
        } else {
            int idx = (int) (System.currentTimeMillis() % areas.size());
            return areas.get(idx);
        }
    }

    /** Simple enhancement utility copied from previous implementation */
    public static final class EnhanceUtils {
        public static Mat enhance(Mat src) {
            Mat blurred = new Mat();
            Imgproc.GaussianBlur(src, blurred, new Size(5, 5), 0);

            Mat enhanced = new Mat();
            if (src.channels() == 1) {
                Imgproc.equalizeHist(blurred, enhanced);
            } else {
                Mat hsv = new Mat();
                Imgproc.cvtColor(blurred, hsv, Imgproc.COLOR_BGR2HSV);
                List<Mat> channels = new ArrayList<>();
                Core.split(hsv, channels);
                Imgproc.equalizeHist(channels.get(2), channels.get(2));
                Core.merge(channels, hsv);
                Imgproc.cvtColor(hsv, enhanced, Imgproc.COLOR_HSV2BGR);
                hsv.release();
                for (Mat c : channels) {
                    c.release();
                }
            }

            Mat sharpened = new Mat();
            Mat kernel = new Mat(3, 3, CvType.CV_32F);
            float[] data = {0, -1, 0, -1, 5, -1, 0, -1, 0};
            kernel.put(0, 0, data);
            Imgproc.filter2D(enhanced, sharpened, -1, kernel);

            blurred.release();
            enhanced.release();
            kernel.release();

            return sharpened;
        }

        private EnhanceUtils() {}
    }





    // You can add your method.
    private String yourMethod(){
        return "your method";
    }



}
