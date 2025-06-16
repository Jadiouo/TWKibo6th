package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import android.util.Log;

import java.util.List;
import java.util.ArrayList;

// OpenCV imports
import org.opencv.aruco.Dictionary;
import org.opencv.aruco.Aruco;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.calib3d.Calib3d;
import org.opencv.imgproc.CLAHE;

public class YourService extends KiboRpcService {

    private final String TAG = this.getClass().getSimpleName();

    @Override
    protected void runPlan1(){
        // Log the start of the mission.
        Log.i(TAG, "Start mission");
        
        // The mission starts.
        api.startMission();

        // Move to a point.
        Point point = new Point(10.9d, -9.92284d, 5.195d);
        Quaternion quaternion = new Quaternion(0f, 0f, -0.707f, 0.707f);
        api.moveTo(point, quaternion, false);

        // Get a camera image.
        Mat image = api.getMatNavCam();

        // ========================================================================
        // CONFIGURABLE IMAGE PROCESSING PARAMETERS - EDIT HERE
        // ========================================================================
        
        Size cropWarpSize = new Size(640, 480);   // Size for cropped/warped image
        Size resizeSize = new Size(320, 320);     // Size for final processing
        
        // Send the image into image enhancement and cropping with custom sizes
        Mat claHeBinImage = imageEnhanceAndCrop(image, cropWarpSize, resizeSize);
        
        if (claHeBinImage != null) {
            Log.i(TAG, "Image enhancement and cropping successful");
            // Now you have the processed binary image ready for YOLO or other detection
            
            // TODO: Add your YOLO detection or other image processing here
            // Example: detectItemsWithYOLO(claHeBinImage);
            
            // Clean up the processed image when done
            claHeBinImage.release();
        } else {
            Log.w(TAG, "Image enhancement failed - no markers detected or processing error");
        }

        // Clean up original image
        image.release();

        /* ******************************************************************************** */
        /* Write your code to recognize the type and number of landmark items in each area! */
        /* If there is a treasure item, remember it.                                        */
        /* ******************************************************************************** */

        // When you recognize landmark items, let's set the type and number.
        api.setAreaInfo(1, "item_name", 1);

        /* **************************************************** */
        /* Let's move to each area and recognize the items. */
        /* **************************************************** */

        // When you move to the front of the astronaut, report the rounding completion.
        point = new Point(11.143d, -6.7607d, 4.9654d);
        quaternion = new Quaternion(0f, 0f, 0.707f, 0.707f);
        api.moveTo(point, quaternion, false);
        api.reportRoundingCompletion();

        /* ********************************************************** */
        /* Write your code to recognize which target item the astronaut has. */
        /* ********************************************************** */

        // Let's notify the astronaut when you recognize it.
        api.notifyRecognitionItem();

        /* ******************************************************************************************************* */
        /* Write your code to move Astrobee to the location of the target item (what the astronaut is looking for) */
        /* ******************************************************************************************************* */

        // Take a snapshot of the target item.
        api.takeTargetItemSnapshot();
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
     * Enhanced image processing method that detects ArUco markers, crops region, 
     * applies CLAHE enhancement, and binarizes the image
     * @param image Input image from NavCam
     * @param cropWarpSize Size for the cropped/warped image (e.g., 640x480)
     * @param resizeSize Size for the final processed image (e.g., 320x320)
     * @return Processed CLAHE + Otsu binarized image, or null if no markers detected
     */
    private Mat imageEnhanceAndCrop(Mat image, Size cropWarpSize, Size resizeSize) {
        try {
            // Save original test image
            api.saveMatImage(image, "test.png");
            
            // Initialize ArUco detection
            Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
            List<Mat> corners = new ArrayList<>();
            Mat ids = new Mat();
            
            // Detect markers
            Aruco.detectMarkers(image, dictionary, corners, ids);
            
            if (corners.size() > 0) {
                Log.i(TAG, "Detected " + corners.size() + " markers.");
                
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
                
                Aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);
                
                // Process first marker only
                Mat imageWithFrame = image.clone();
                Aruco.drawDetectedMarkers(imageWithFrame, corners, ids);
                
                if (rvecs.rows() > 0 && tvecs.rows() > 0) {
                    Mat rvec = new Mat(3, 1, CvType.CV_64F);
                    Mat tvec = new Mat(3, 1, CvType.CV_64F);
                    
                    rvecs.row(0).copyTo(rvec);
                    tvecs.row(0).copyTo(tvec);
                    
                    // Convert to RGB and draw axis
                    Imgproc.cvtColor(imageWithFrame, imageWithFrame, Imgproc.COLOR_GRAY2RGB);
                    Aruco.drawAxis(imageWithFrame, cameraMatrix, distCoeffs, rvec, tvec, 0.1f);
                    
                    // Save marker with frame
                    api.saveMatImage(imageWithFrame, "marker_0_with_frame.png");
                    
                    // Process crop region and return enhanced image with custom sizes
                    Mat processedImage = processCropRegion(image, cameraMatrix, distCoeffs, rvec, tvec, cropWarpSize, resizeSize);
                    
                    // Clean up
                    rvec.release();
                    tvec.release();
                    imageWithFrame.release();
                    cameraMatrix.release();
                    distCoeffs.release();
                    rvecs.release();
                    tvecs.release();
                    
                    // Clean up corners and ids
                    ids.release();
                    for (Mat corner : corners) {
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
            } else {
                Log.w(TAG, "No ArUco markers detected in image");
            }
            
            // Clean up
            ids.release();
            for (Mat corner : corners) {
                corner.release();
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
    private Mat processCropRegion(Mat image, Mat cameraMatrix, Mat distCoeffs, Mat rvec, Mat tvec, Size cropWarpSize, Size resizeSize) {
        try {
            // Define crop area corners in 3D (manually adjusted)
            org.opencv.core.Point3[] cropCorners3D = {
                new org.opencv.core.Point3(-0.0325, 0.0375, 0),    // Top-left
                new org.opencv.core.Point3(-0.2325, 0.0375, 0),   // Top-right  
                new org.opencv.core.Point3(-0.2325, -0.1125, 0),  // Bottom-right
                new org.opencv.core.Point3(-0.0325, -0.1125, 0)   // Bottom-left
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
                Mat processedImage = cropEnhanceAndBinarize(image, cropPoints2D, cropWarpSize, resizeSize);
                
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
     */
    private Mat cropEnhanceAndBinarize(Mat image, org.opencv.core.Point[] cropPoints2D, Size cropWarpSize, Size resizeSize) {
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
            
            // Save the cropped image with dynamic filename
            String cropFilename = String.format("cropped_region_%.0fx%.0f.png", cropWarpSize.width, cropWarpSize.height);
            api.saveMatImage(croppedImage, cropFilename);
            Log.i(TAG, "Cropped region saved as " + cropFilename);
            
            // ========================================================================
            // STEP 2: Resize to final processing size (configurable)
            // ========================================================================
            
            // Resize the cropped image to final size
            Mat resizedImage = new Mat();
            Imgproc.resize(croppedImage, resizedImage, resizeSize);
            
            // Save resized image
            String resizeFilename = String.format("yolo_original_%.0fx%.0f.png", resizeSize.width, resizeSize.height);
            api.saveMatImage(resizedImage, resizeFilename);
            Log.i(TAG, "Resized image saved as " + resizeFilename);
            
            // ========================================================================
            // STEP 3: Apply CLAHE enhancement
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
            
            // ========================================================================
            // STEP 4: Apply Otsu's binarization
            // ========================================================================
            
            // Apply Otsu's automatic threshold binarization
            Mat binarizedOtsu = new Mat();
            double otsuThreshold = Imgproc.threshold(claheImage, binarizedOtsu, 0, 255, 
                                                Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
            
            // Print min/max values and threshold of Otsu binarized image
            Core.MinMaxLocResult binaryOtsuResult = Core.minMaxLoc(binarizedOtsu);
            Log.i(TAG, String.format("Binary Otsu (%.1f) - Min: %.2f, Max: %.2f", 
                otsuThreshold, binaryOtsuResult.minVal, binaryOtsuResult.maxVal));
            
            // Save the Otsu binarized image with dynamic filename
            String binaryFilename = String.format("yolo_binary_otsu_%.0fx%.0f.png", resizeSize.width, resizeSize.height);
            api.saveMatImage(binarizedOtsu, binaryFilename);
            Log.i(TAG, String.format("Otsu binary image saved as %s (threshold: %.1f)", binaryFilename, otsuThreshold));
            
            // ========================================================================
            // CLEANUP
            // ========================================================================
            
            // Clean up intermediate images
            srcPointsMat.release();
            dstPointsMatCrop.release();
            perspectiveMatrixCrop.release();
            croppedImage.release();
            resizedImage.release();
            claheImage.release();
            
            // Return the final processed binary image
            return binarizedOtsu;
            
        } catch (Exception e) {
            Log.e(TAG, "Error in cropEnhanceAndBinarize: " + e.getMessage());
            return null;
        }
    }

    // You can add your method.
    private String yourMethod(){
        return "your method";
    }
}