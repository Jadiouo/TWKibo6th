package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

//import org.opencv.core.Mat;

// new imports
import android.util.Log;

import java.util.List;
import java.util.ArrayList;


// new OpenCV imports
import org.opencv.aruco.Dictionary;
import org.opencv.aruco.Aruco;
import org.opencv.core.*;
// OpenCV Core imports
//import org.opencv.core.CvType;
//import org.opencv.core.Scalar;
//import org.opencv.core.Rect;
//import org.opencv.core.Point3;
//import org.opencv.core.MatOfPoint3f;
//import org.opencv.core.MatOfPoint2f;
//import org.opencv.core.MatOfDouble;

// OpenCV Image Processing
import org.opencv.imgproc.Imgproc;

// OpenCV Calibration
import org.opencv.calib3d.Calib3d;
import gov.nasa.arc.astrobee.Result;
/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 */

public class YourService extends KiboRpcService {

    // The TAG is used for logging.
    // You can use it to check the log in the Android Studio.
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
        // Save the image to a file.
        api.saveMatImage(image, "test.png");


        /* ******************************************************************************** */
        /* Write your code to recognize the type and number of landmark items in each area! */
        /* If there is a treasure item, remember it.                                        */
        /* ******************************************************************************** */

        // 
        /**
         * Retrieves a predefined Aruco dictionary for 6x6 markers containing 250 distinct patterns.
         * This dictionary is used for detecting and tracking Aruco markers in images.
         *
         * The call to Aruco.getPredefinedDictionary(Aruco.DICT_6X6_250) selects a standard set of marker patterns,
         * making it easier to consistently identify markers during image processing.
         */
        Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        
        // Detect markers in the image using the specified dictionary.
        // The detectMarkers function analyzes the image and identifies the locations of Aruco markers.
        // The detected markers are stored in the corners list.
        // The corners list contains the coordinates of the detected markers in the image.
        
        List<Mat> corners = new ArrayList<>();
        Mat ids = new Mat();
        // The ids list contains the IDs of the detected markers.
        Aruco.detectMarkers(image, dictionary, corners, ids);

        if (corners.size() > 0) {
            // Get camera parameters
            double[][] intrinsics = api.getNavCamIntrinsics();
            Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
            Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);
            
            // Fill camera matrix
            cameraMatrix.put(0, 0, intrinsics[0]);
            distCoeffs.put(0, 0, intrinsics[1]);
            distCoeffs.convertTo(distCoeffs, CvType.CV_64F);

            
            // Estimate pose for each marker
            Mat rvecs = new Mat();
            Mat tvecs = new Mat();
            
            // Marker size in meters (adjust according to your actual marker size)
            float markerLength = 0.05f; // 5cm markers
            
            Aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);
            
            // Draw markers and coordinate frames
            Mat imageWithFrame = image.clone();
            Mat imagewithcroparea = image.clone();

            Aruco.drawDetectedMarkers(imageWithFrame, corners, ids);
            
            // type the number of corners in log
            Log.i(TAG, "Detected " + corners.size() + " markers.");
            
            // Draw coordinate frame for each marker
            for (int i = 0; i < corners.size(); i++) {
                
                // extract the corners name variable "currentCorners"
                Mat currentCorners = corners.get(i);
                

                Mat UndistortImg = new Mat();

                if (rvecs.rows() > 0 && tvecs.rows() > 0) {
                    Mat rvec = new Mat(3, 1, CvType.CV_64F);
                    Mat tvec = new Mat(3, 1, CvType.CV_64F);
                    
                    rvecs.row(i).copyTo(rvec);
                    tvecs.row(i).copyTo(tvec);
                    
                    Imgproc.cvtColor(imageWithFrame, imageWithFrame, Imgproc.COLOR_GRAY2RGB);
                    // Simply draw the coordinate frame
                    Aruco.drawAxis(imageWithFrame, cameraMatrix, distCoeffs, rvec, tvec, 0.1f); // 0.1m axis length

                    // corners.get(i) is Mat we need to convert it to MatOfPoint2f
                    MatOfPoint2f cornerPoints = new MatOfPoint2f(currentCorners);

                    // check corner[0] and crop region
                    checkCornerAndCropRegion(imagewithcroparea, cameraMatrix, distCoeffs, rvec, tvec, cornerPoints);
                    api.saveMatImage(imagewithcroparea, "marker_" + i + "_crop_area.png");
                    // drawCropArea(imagewithcroparea, corners.get(i).toArray());


                    // Release individual vectors
                    cornerPoints.release();
                    rvec.release();
                    tvec.release();
                }


                // Extract rotation and translation vectors for this marker
                // rvecs.row(i).copyTo(rvec);
                // tvecs.row(i).copyTo(tvec);

                // Draw coordinate frame (X=red, Y=green, Z=blue)
                // drawCoordinateFrame(imageWithFrame, cameraMatrix, distCoeffs, rvec, tvec, markerLength);
                
                // Crop region around the marker
                Mat croppedRegion = cropMarkerRegion(image, corners.get(i));
                
                // Save cropped image for debugging
                api.saveMatImage(croppedRegion, "marker_" + i + "_cropped.png"); 
                // see what color type of imageWithFrame is
                Log.i(TAG, "Image with frame type: " + imageWithFrame.type());

                // make imageWithFrame from GRAY to RGB
                
                // Save image with drawn frames
                api.saveMatImage(imageWithFrame, "marker_" + i + "_with_frame.png");

                
                Calib3d.undistort(image, UndistortImg, cameraMatrix, distCoeffs);

                // Save undistorted image for debugging
                api.saveMatImage(UndistortImg, "marker_" + i + "_undistorted.png");


                // Clean up individual vectors
                // rvec.release();
                // tvec.release();
                croppedRegion.release();
                UndistortImg.release();
                imageWithFrame.release();

            }

            // Clean up
            rvecs.release();
            tvecs.release();
            // UndistortImg.release();
            cameraMatrix.release();
            distCoeffs.release();
        }

        // Clean up
        ids.release();
        for (Mat corner : corners) {
            corner.release();
        }
        
        
        // When you recognize landmark items, let’s set the type and number.
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



    private void checkCornerAndCropRegion(Mat image, Mat cameraMatrix, Mat distCoeffs, 
                                        Mat rvec, Mat tvec, MatOfPoint2f corners) {
        try {
            // 1. Check if corner[0] is close to expected 3D position (-0.025, 0.025, 0)
            org.opencv.core.Point[] cornerPoints = corners.toArray();
            if (cornerPoints.length > 0) {
                // Project expected 3D point to 2D
                org.opencv.core.Point3[] expectedPoint3D = {new org.opencv.core.Point3(-0.025, 0.025, 0)};
                MatOfPoint3f expectedPointMat = new MatOfPoint3f(expectedPoint3D);
                MatOfPoint2f projectedExpected = new MatOfPoint2f();
                
                // Convert distortion coefficients
                double[] distData = new double[5];
                distCoeffs.get(0, 0, distData);
                MatOfDouble distCoeffsDouble = new MatOfDouble();
                distCoeffsDouble.fromArray(distData);
                
                // Project expected point
                Calib3d.projectPoints(expectedPointMat, rvec, tvec, cameraMatrix, distCoeffsDouble, projectedExpected);
                org.opencv.core.Point[] projectedPoints = projectedExpected.toArray();
                
                if (projectedPoints.length > 0) {
                    org.opencv.core.Point expectedCorner = projectedPoints[0];
                    org.opencv.core.Point actualCorner = cornerPoints[0];
                    
                    // Calculate distance between expected and actual corner
                    // double distance = Math.sqrt(Math.pow(expectedCorner.x - actualCorner.x, 2) + 
                    //                         Math.pow(expectedCorner.y - actualCorner.y, 2));
                    
                    double distance = getdistance(expectedCorner, actualCorner);
                    // Check if close (within 10 pixels tolerance)
                    if (distance > 10.0) {
                        // Log the real 3D position of corner[0]
                        // To get 3D position, we need to reverse project (assuming z=0 for simplicity)
                        Log.i(TAG, String.format("Corner[0] not at expected position. Detected at 2D: (%.2f, %.2f), Expected 2D: (%.2f, %.2f), Distance: %.2f pixels", 
                                actualCorner.x, actualCorner.y, expectedCorner.x, expectedCorner.y, distance));
                    } else {
                        Log.i(TAG, "Corner[0] is close to expected position (-2.5, 2.5, 0)");
                    }
                }
                
                // Clean up
                expectedPointMat.release();
                projectedExpected.release();
                distCoeffsDouble.release();
            }
            
            // 2. Project the 4 crop area corners to 2D
            // Manually adjustment
            org.opencv.core.Point3[] cropCorners3D = {
                new org.opencv.core.Point3(-0.0325, 0.0375, 0),    // Top-left
                new org.opencv.core.Point3(-0.2325, 0.0375, 0),   // Top-right  
                new org.opencv.core.Point3(-0.2325, -0.1125, 0), // Bottom-right
                new org.opencv.core.Point3(-0.0325, -0.1125, 0)   // Bottom-left
            };
            
            MatOfPoint3f cropCornersMat = new MatOfPoint3f(cropCorners3D);
            MatOfPoint2f cropCorners2D = new MatOfPoint2f();
            
            // Convert distortion coefficients again
            double[] distData = new double[5];
            distCoeffs.get(0, 0, distData);
            MatOfDouble distCoeffsDouble = new MatOfDouble();
            distCoeffsDouble.fromArray(distData);
            
            // Project crop corners to 2D
            Calib3d.projectPoints(cropCornersMat, rvec, tvec, cameraMatrix, distCoeffsDouble, cropCorners2D);
            org.opencv.core.Point[] cropPoints2D = cropCorners2D.toArray();
            
            if (cropPoints2D.length == 4) {
                // 3. Create perspective transformation and crop
                cropAndSaveRegion(image, cropPoints2D);
            }
            
            // Clean up
            cropCornersMat.release();
            cropCorners2D.release();
            distCoeffsDouble.release();
            
        } catch (Exception e) {
            Log.e(TAG, "Error in checkCornerAndCropRegion: " + e.getMessage());
        }
    }

    private void cropAndSaveRegion(Mat image, org.opencv.core.Point[] cropPoints2D) {
        try {
            // Define destination points for 640x480 rectangle
            org.opencv.core.Point[] dstPoints = {
                new org.opencv.core.Point(0, 0),       // Top-left
                new org.opencv.core.Point(639, 0),     // Top-right
                new org.opencv.core.Point(639, 479),   // Bottom-right
                new org.opencv.core.Point(0, 479)      // Bottom-left
            };
            
            // Create source and destination point matrices
            MatOfPoint2f srcPointsMat = new MatOfPoint2f(cropPoints2D);
            MatOfPoint2f dstPointsMat = new MatOfPoint2f(dstPoints);
            
            // Calculate perspective transformation matrix
            Mat perspectiveMatrix = Imgproc.getPerspectiveTransform(srcPointsMat, dstPointsMat);
            
            // Apply perspective transformation
            Mat croppedImage = new Mat();
            Imgproc.warpPerspective(image, croppedImage, perspectiveMatrix, new Size(640, 480));
            
            // Save the cropped image
            api.saveMatImage(croppedImage, "cropped_region_640x480.png");
            Log.i(TAG, "Cropped region saved as 640x480 image");
            
            // Optional: Draw the crop area on original image for visualization
            drawCropArea(image, cropPoints2D);
            
            // Clean up
            srcPointsMat.release();
            dstPointsMat.release();
            perspectiveMatrix.release();
            croppedImage.release();
            
        } catch (Exception e) {
            Log.e(TAG, "Error cropping region: " + e.getMessage());
        }
    }

    // ------------------------------------------------------------------------------------

    //-------------------------------------------------------------------------------------

    /**
     * Crops a region around the detected marker with some padding
     */
    private Mat cropMarkerRegion(Mat image, Mat markerCorners) {
        // Get the four corners of the marker
        float[] cornerData = new float[(int)(markerCorners.total() * markerCorners.channels())];
        markerCorners.get(0, 0, cornerData);
        
        // Find bounding rectangle
        float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE;
        float maxX = Float.MIN_VALUE, maxY = Float.MIN_VALUE;
        
        for (int i = 0; i < cornerData.length; i += 2) {
            float x = cornerData[i];
            float y = cornerData[i + 1];
            
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
        }
        
        // Add padding (20% of marker size)
        float padding = Math.max(maxX - minX, maxY - minY) * 0.2f;
        
        int x = Math.max(0, (int)(minX - padding));
        int y = Math.max(0, (int)(minY - padding));
        int width = Math.min(image.cols() - x, (int)(maxX - minX + 2 * padding));
        int height = Math.min(image.rows() - y, (int)(maxY - minY + 2 * padding));
        
        // Create rectangle and crop
        Rect cropRect = new Rect(x, y, width, height);
        Mat croppedImage = new Mat(image, cropRect);
        
        return croppedImage.clone();
    }

    private void drawCropArea(Mat image, org.opencv.core.Point[] cropPoints2D) {
        // turn the image to RGB
        Imgproc.cvtColor(image, image, Imgproc.COLOR_GRAY2RGB);
        // Draw the crop area outline on the original image
        for (int i = 0; i < cropPoints2D.length; i++) {
            org.opencv.core.Point pt1 = cropPoints2D[i];
            org.opencv.core.Point pt2 = cropPoints2D[(i + 1) % cropPoints2D.length];
            Imgproc.line(image, pt1, pt2, new Scalar(255, 255, 0), 2); // Yellow lines
            // input to log
            Log.i(TAG, String.format("Crop Point %d: (%.2f, %.2f)", i, pt1.x, pt1.y));
        }
        
        // Add corner labels
        for (int i = 0; i < cropPoints2D.length; i++) {
            Imgproc.putText(image, String.valueOf(i), cropPoints2D[i], 
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 255, 0), 2);
        }
    }

    private Result sureMoveToPoint(Point point, Quaternion quaternion, boolean printRobotPosition, int maxRetries) {
        Result result = api.moveTo(point, quaternion, printRobotPosition);
        
        int retryCount = 0;
        while (!result.hasSucceeded() && retryCount < maxRetries) {
            result = api.moveTo(point, quaternion, true); // Use true for retries
            retryCount++;
        }
        
        return result;
    }


    private double getdistance(org.opencv.core.Point p1, org.opencv.core.Point p2) {
        // Calculate the distance between two points using the Euclidean distance formula
        return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
    }

    // You can add your method.
    private String yourMethod(){
        return "your method";
    }
}
