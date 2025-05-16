# method 2
#!/usr/bin/env python

import cv2
import numpy as np
from scipy.interpolate import CubicSpline

class LaneDetector:
    def __init__(self):
        # self.bridge = CvBridge()
        # self.image_sub = rospy.Subscriber("/carla/ego_vehicle/camera", Image, self.image_callback)
        
        # Define IPM parameters (adjust these based on your camera setup)
        #town 04
        # self.src_points = np.float32([
        #     [0, 300],  # bottom-left
        #     [800, 300],  # bottom-right
        #     [480, 40],  # top-right
        #     [350, 40]   # top-left
        # ])
        
        # self.dst_points = np.float32([
        #     [150, 1500],  # bottom-left
        #     [850, 1500],  # bottom-right
        #     [1000, 700],    # top-right
        #     [150, 700]     # top-left
        # ])
        #town 05
        self.src_points = np.float32([
            [230, 370],  # bottom-left
            [1053, 370],  # bottom-right
            [650, 20],  # top-right
            [625, 20]   # top-left
        ])
        
        self.dst_points = np.float32([
            [150, 370],  # bottom-left
            [1000, 370],  # bottom-right
            [1000, 0],    # top-right
            [150, 0]     # top-left
        ])
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.count = 0
        # self.M = np.float32([[-0.1875, 0, 75], 
        #                     [0, 0.16666667, -100],
        #                     [0, -0.01666667, 1]])

    def sobel_edge_detection(self, img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Sobel operators
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        magnitude = np.sqrt(abs_sobelx**2 + abs_sobely**2)
        
        # Normalize and threshold
        scaled_sobel = np.uint8(255 * magnitude / np.max(magnitude))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= 100)] = 1

        # cv2.imshow("soble_image", binary_output)
        # cv2.waitKey(1)
        
        return binary_output
    
    def hls_transform(self, img):
        lower_white = np.array([0, 180, 0])
        upper_white = np.array([255, 255, 30])
        # Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        
        # Define thresholds for S channel (Saturation)
        # These values might need adjustment based on your specific conditions
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        l_channel=l_channel*(255/np.max(l_channel))
        # s_channel=s_channel*(255/np.max(s_channel))
        l_thresh_min = 210 #np.mean(l_channel)#200
        l_thresh_max = 255
        s_thresh_min = 0
        s_thresh_max = 150
        
        # Create binary image from S channel
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel > l_thresh_min)] = 1

        # s_binary = np.zeros_like(s_channel)
        # s_binary[(s_thresh_min < s_channel) & (s_channel < s_thresh_max)] = 1
        # w_binary = np.zeros_like(l_binary)
        # w_binary [(l_binary == 1) & (s_binary == 1)] = 1
        # cv2.imshow("hls_image", l_binary)
        # cv2.waitKey(1)
    
        return l_binary

    def apply_ipm(self, img):
        # Apply Inverse Perspective Mapping
        warped = cv2.warpPerspective(img, self.M, (1200, 370))
        # cv2.imshow("image", img)
        # cv2.imshow("ipm_image", warped)
        # cv2.waitKey(1)
        return warped

    def extract_trapezoid_to_image(self, input_image, trapezoid_points):
        """
        Extract a trapezoid region from an image and save it as a new image.
        
        Parameters:
        - input_image_path: Path to the input image
        - output_image_path: Path to save the extracted trapezoid image
        - trapezoid_points: List of 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] 
                        defining the trapezoid in clockwise or counter-clockwise order
        """
        # Load the image
            # Convert points to numpy array
        pts = np.array(trapezoid_points, dtype=np.int32)

        # Create a mask with the same size as the image
        mask = np.zeros_like(input_image, dtype=np.uint8)
        
        # Fill the trapezoid region in the mask with white (255)
        cv2.fillPoly(mask, [pts], (255, 255, 255))

        # Apply the mask to the original image
        extracted = cv2.bitwise_and(input_image, mask)

        return extracted

    def bezier_curve(self, control_points, num_points=100):
        """Generate points on a third-order Bézier curve given 4 control points."""
        t = np.linspace(0, 1, num_points)
        curve_points = np.zeros((num_points, 2))
        
        for i in range(num_points):
            tt = t[i]
            # Third-order Bézier curve equation
            curve_points[i] = (
                (1-tt)**3 * control_points[0] +
                3 * (1-tt)**2 * tt * control_points[1] +
                3 * (1-tt) * tt**2 * control_points[2] +
                tt**3 * control_points[3]
            )
        
        return curve_points.astype(np.int32)

    def fit_spline(self, centers, num_points=100, img_height=1500):
        # """Fit a parametric cubic spline to a set of points."""
        if len(centers) < 2:  # Need at least 2 points for a spline
            return None
        
        centers = np.array(centers, dtype=np.float32)
        n = len(centers)
        
        # Use normalized y-coordinates as parameter t
        y = centers[:, 1]
        t = (y - y.min()) / (y.max() - y.min()) if y.max() != y.min() else np.linspace(0, 1, n)
        
        # Ensure t is strictly increasing (required by CubicSpline)
        if np.any(np.diff(t) <= 0):
            t = np.linspace(0, 1, n)
        
        # Fit cubic splines for x and y coordinates
        x_spline = CubicSpline(t, centers[:, 0])
        y_spline = np.linspace(img_height, 0, num_points)
        
        # Generate points along the spline
        t_new = np.linspace(0, 1, num_points)
        spline_points = np.column_stack((x_spline(t_new), y_spline))
        return spline_points.astype(np.int32)
        # if len(centers) < 2:  # Need at least 2 points for a spline
        #     return None
        
        # centers = np.flip(np.array(centers, dtype=np.float32), axis=0)
        
        # # Fit cubic splines for x and y coordinates
        # cs = CubicSpline(centers[:, 1], centers[:, 0])
        # y_spline = np.linspace(0, 1500, num_points)
        
        # # Generate points along the spline
        # spline_points = np.column_stack((cs(y_spline), y_spline))
        
        # return np.flip(spline_points.astype(np.int32), axis = 0)

    def check_turn_left_or_right(self, lane_windows):
        curve_right = 0
        curve_left = 0
        for i in range(1, len(lane_windows)):
            if lane_windows[i][0][0] == lane_windows[i - 1][0][0]:
                continue
            if lane_windows[i][2][0] < lane_windows[i - 1][2][0]:
                curve_right += 1
            else:
                curve_left += 1

        if curve_right - curve_left > len(lane_windows)//2: #turn left
            return 1, 0
        
        if curve_right - curve_left < len(lane_windows)//2: # turn right
            return 0, 1
        
        return 0, 0 #straight

    def sliding_windows(self, binary_warped, nwindows=60, margin=40, minpix=100, maxpix=1000, move_step = 10, move_num = 3):
        # Create an output image to draw on
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        out_img = out_img.astype(np.uint8)  # Explicitly convert to uint8
        
        # Find the histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        
        # Find the peak of the left and right halves
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]//nwindows)
        
        # Identify the x and y positions of all nonzero pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        left_lane_x_inds = []
        lane_y_inds = []
        right_lane_x_inds = []
        left_lane_inds_np = []
        right_lane_inds_np = []
        windows = []  # Store window coordinates
        left_centers = []
        right_centers = []

        all_windows = np.zeros((nwindows, 6))
        plot_left = []
        plot_right = []
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Identify nonzero pixels within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                             (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Recenter next window if enough pixels found
            find_left = 0
            find_right = 0

            cur_best_fit_left = leftx_current
            cur_best_fit_right = rightx_current

            window_move_left = 0
            window_move_right = 0
            if len(windows) >= 3:
                curve_left, curve_right = self.check_turn_left_or_right(windows)

                if curve_left:
                    window_move_left = 5
                    window_move_right = -5
                elif curve_right:
                    window_move_left = -5
                    window_move_right = 5

            for i in range(-(move_num + window_move_left), move_num + window_move_right + 1):
                win_xleft_low = leftx_current - margin + i * move_step
                win_xleft_high = leftx_current + margin + i * move_step
                win_xright_low = rightx_current - margin + i * move_step
                win_xright_high = rightx_current + margin + i * move_step
                tem_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                                (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                tem_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

                if len(tem_left_inds) > len(good_left_inds):
                    cur_best_fit_left = leftx_current + i* move_step
                    good_left_inds = tem_left_inds
                    # find_left = 1
                if len(tem_right_inds) > len(good_right_inds):
                    cur_best_fit_right = rightx_current  + i * move_step
                    good_right_inds = tem_right_inds
                    # find_right = 1

            leftx_current = cur_best_fit_left
            rightx_current = cur_best_fit_right

            if minpix < len(good_left_inds) < maxpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                find_left = 1
            if minpix < len(good_right_inds) < maxpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                find_right = 1

            if find_left and find_right and (rightx_current - leftx_current) < 650:
                continue

            if find_left == 0 and find_right == 0:
                continue

            if find_left == 1 and find_right == 0:
                rightx_current = leftx_current + 852
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                # print("left number: ", len(good_left_inds))
            elif find_left == 0 and find_right == 1:
                leftx_current = rightx_current - 852
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                                (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                # print("right number: ", len(good_right_inds))
            # print("left right: ", len(good_left_inds), len(good_right_inds) )
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            if find_left == 1 or find_right == 1:
                windows.append([(win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                            (win_xright_low, win_y_low), (win_xright_high, win_y_high)])
                # if find_left:
                left_centers.append([win_xleft_low , win_y_low ])
                # if find_right:
                right_centers.append([win_xright_low, win_y_low])
                all_windows[window][0] = (win_xleft_low + win_xleft_high) / 2
                all_windows[window][1] = win_y_low
                all_windows[window][2] = 1
                all_windows[window][3] = (win_xright_low + win_xright_high) / 2
                all_windows[window][4] = win_y_high
                all_windows[window][5] = 1
                # Draw the windows on the visualization image
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), 
                            (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), 
                            (win_xright_high, win_y_high), (0, 255, 0), 2)
        
        left_fitting_points = None
        right_fitting_points = None
        if len(left_centers) >= 4:
            # Select 4 control points (first, two middle, last)
            left_centers = np.array(left_centers)
            left_window_num = (left_centers[0][1]- left_centers[-1][1])//window_height
            indices = np.linspace(0, len(left_centers)-1, 4, dtype=int)
            control_points = left_centers[indices]
            left_fitting_points = self.bezier_curve(control_points, left_window_num)
            # left_fitting_points = self.fit_spline(left_centers, nwindows, binary_warped.shape[0])
            # Draw left Bézier curve
            for i in range(1, len(left_fitting_points)):
                cv2.line(out_img, tuple(left_fitting_points[i-1]), tuple(left_fitting_points[i]), (0, 255, 255), 4)
        
        if len(right_centers) >= 4:
            right_centers = np.array(right_centers)
            right_window_num = (right_centers[0][1] - right_centers[-1][1])//window_height
            indices = np.linspace(0, len(right_centers)-1, 4, dtype=int)
            control_points = right_centers[indices]
            right_fitting_points = self.bezier_curve(control_points, right_window_num)
            # right_fitting_points = self.fit_spline(right_centers, nwindows, binary_warped.shape[0])
            # Draw right Bézier curve
            for i in range(1, len(right_fitting_points)):
                cv2.line(out_img, tuple(right_fitting_points[i-1]), tuple(right_fitting_points[i]), (0, 255, 255), 4)

        if left_fitting_points is not None:
            for i in range(len(left_fitting_points)):
                win_xleft_low, win_y_low = left_fitting_points[i]
                
                win_xleft_high = win_xleft_low + margin * 2
                # win_y_low = binary_warped.shape[0] - i * window_height
                win_y_high = win_y_low - window_height
                # win_y_high = binary_warped.shape[0] - (i + 1) * window_height
                # win_xright_low, _ = right_fitting_points[i]
                # win_xright_high = win_xright_low + margin * 2
                # if all_windows[i][2] == 0:
                #     all_windows[i][0] = (win_xleft_low + win_xleft_high) / 2
                #     all_windows[i][1] = win_y_low
                #     all_windows[i][2] = 1
                #     all_windows[i][3] = (win_xright_low + win_xright_high) / 2
                #     all_windows[i][4] = win_y_high
                #     all_windows[i][5] = 1

                # Draw the windows on the visualization image
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), 
                            (win_xleft_high, win_y_high), (255, 0, 0), 2)
                # cv2.rectangle(out_img, (win_xright_low, win_y_low), 
                #             (win_xright_high, win_y_high), (255, 0, 0), 2)
                
            for i in range(len(right_fitting_points)):
                # win_xleft_low, win_y_low = left_fitting_points[i]
                
                # win_xleft_high = win_xleft_low + margin * 2
                # win_y_low = binary_warped.shape[0] - i * window_height
                # win_y_high = binary_warped.shape[0] - (i + 1) * window_height
                win_xright_low, win_y_low = right_fitting_points[i]
                # win_y_low = binary_warped.shape[0] - i * window_height
                win_y_high = win_y_low - window_height
                win_xright_high = win_xright_low + margin * 2
                # if all_windows[i][2] == 0:
                #     all_windows[i][0] = (win_xleft_low + win_xleft_high) / 2
                #     all_windows[i][1] = win_y_low
                #     all_windows[i][2] = 1
                #     all_windows[i][3] = (win_xright_low + win_xright_high) / 2
                #     all_windows[i][4] = win_y_high
                #     all_windows[i][5] = 1

                # Draw the windows on the visualization image
                # cv2.rectangle(out_img, (win_xleft_low, win_y_low), 
                #             (win_xleft_high, win_y_high), (255, 0, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), 
                            (win_xright_high, win_y_high), (255, 0, 0), 2)

        #     win_xleft_low = leftx_current - margin
        #     win_xleft_high = leftx_current + margin
        #     win_xright_low = rightx_current - margin
        #     win_xright_high = rightx_current + margin
        #     all_windows[window][0] = (win_xleft_low + win_xleft_high) / 2
        #     all_windows[window][1] = win_y_high
        #     all_windows[window][2] = 1
        #     all_windows[window][3] = (win_xright_low + win_xright_high) / 2
        #     all_windows[window][4] = win_y_high
        #     all_windows[window][5] = 1
        #     plot_left.append([(win_xleft_low + win_xleft_high) / 2, win_y_high])
        #     plot_right.append([(win_xright_low + win_xright_high) / 2, win_y_high])

        # left_fit = np.polyfit(plot_left[1], plot_left[0], 2)
        # right_fit = np.polyfit(plot_right[1], plot_right[0], 2)

        # for i in range(nwindows):
        #     win_y_low = binary_warped.shape[0] - (i + 1) * window_height
        #     win_y_high = binary_warped.shape[0] - i * window_height
        #     leftx_current = all_windows[i][0]
        #     rightx_current = all_windows[i][3]
        #     if all_windows[i][2] == 0:
        #         leftx_current = left_fit[0] * win_y_high**2 + left_fit[1] * win_y_high + left_fit[2]
        #         rightx_current = right_fit[0] * win_y_high**2 + right_fit[1] * win_y_high + right_fit[2]
        #     win_xleft_low = int(leftx_current - margin)
        #     win_xleft_high = int(leftx_current + margin)
        #     win_xright_low = int(rightx_current - margin)
        #     win_xright_high = int(rightx_current + margin)

        #     windows.append([(win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
        #                 (win_xright_low, win_y_low), (win_xright_high, win_y_high)])
        
        #     # Draw the windows on the visualization image
        #     cv2.rectangle(out_img, (win_xleft_low, win_y_low), 
        #                 (win_xleft_high, win_y_high), (0, 255, 0), 2)
        #     cv2.rectangle(out_img, (win_xright_low, win_y_low), 
        #                 (win_xright_high, win_y_high), (0, 255, 0), 2)

        
            # Append these indices method 1
            # left_lane_inds.append(good_left_inds)
            # right_lane_inds.append(good_right_inds)
            
            # if left_lane_inds_np is not None:
            #     left_lane_inds_np = np.concatenate(left_lane_inds)
            #     lefttx=nonzerox[left_lane_inds_np]
            #     leftty=nonzeroy[left_lane_inds_np]
            #     if (len(leftty) != 0)  and (len(lefttx) != 0):
            #         left_fit = np.polyfit(leftty, lefttx, 2)

            #         ploty = win_y_low + window_height / 2
            #         # left_fitx=left_fit[0] * ploty**3 + left_fit[1] * ploty**2 + ploty * left_fit[2] + left_fit[3]
            #         leftx_current = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            
            # if right_lane_inds_np is not None:
            #     # Concatenate the arrays of indices
            #     right_lane_inds_np = np.concatenate(right_lane_inds)
            #     righttx=nonzerox[right_lane_inds_np]
            #     rightty=nonzeroy[right_lane_inds_np]
            #     if (len(righttx) != 0)  and (len(rightty) != 0):
            #         right_fit = np.polyfit(rightty, righttx, 2)

            #         ploty = win_y_low + window_height / 2
            #         # right_fitx=right_fit[0] * ploty**3 + right_fit[1] * ploty**2 + ploty * right_fit[2] + right_fit[3]
            #         rightx_current = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

            # # Append these indices method 2
            # left_lane_x_inds.append(leftx_current)
            # lane_y_inds.append(win_y_low + window_height / 2)
            # right_lane_x_inds.append(good_right_inds)
            
            # if left_lane_inds_np is not None:
            #     if (len(left_lane_x_inds) != 0)  and (len(lane_y_inds) != 0):
            #         left_fit = np.polyfit(lane_y_inds, left_lane_x_inds, 2)

            #         ploty = win_y_low - window_height / 2
            #         # left_fitx=left_fit[0] * ploty**3 + left_fit[1] * ploty**2 + ploty * left_fit[2] + left_fit[3]
            #         leftx_current = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            
            # if right_lane_inds_np is not None:
            #     # Concatenate the arrays of indices
            #     if (len(right_lane_x_inds) != 0)  and (len(lane_y_inds) != 0):
            #         right_fit = np.polyfit(lane_y_inds, right_lane_x_inds, 2)

            #         ploty = win_y_low - window_height / 2
            #         # right_fitx=right_fit[0] * ploty**3 + right_fit[1] * ploty**2 + ploty * right_fit[2] + right_fit[3]
            #         rightx_current = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        # cv2.imwrite("/home/ran/envs/CARLA_0.9.13/myProject/data/lane_" + str(self.count) + ".jpg", out_img)
        return all_windows, out_img, windows
    
    def combine_image(self, hls_binary, sobel_binary):
        combine = np.zeros_like(hls_binary)
        combine[(hls_binary > 0) & (sobel_binary > 0)] = 1
        # cv2.imwrite("/home/ran/envs/CARLA_0.9.13/myProject/test_img/hls_" + str(self.count) + ".jpg", hls_binary * 255)
        # cv2.imwrite("/home/ran/envs/CARLA_0.9.13/myProject/test_img/sobel_" + str(self.count) + ".jpg", sobel_binary * 255)
        return combine

    def get_white_part(self, image, r_thres = 200, g_thres = 200, b_thres = 200):
        
        r_channel = image[:,:,0]
        g_channel = image[:,:,1]
        b_channel = image[:,:,2]
        white_part = np.zeros_like(b_channel)
        white_part[(r_channel > r_thres) & (g_channel> g_thres) & (b_channel > b_thres)] = 1
        return white_part

    def lane_detect(self, image):
        #town 04
        cv_image = image[350 :, :, : ]
        # cv2.imwrite("D:\Apps\carla\CARLA_0.9.14\WindowsNoEditor\image\ori_" + str(self.count) + ".jpg", cv_image)

        # Apply IPM
        warped = self.apply_ipm(cv_image)
        # cv2.imwrite("D:\Apps\carla\CARLA_0.9.14\WindowsNoEditor\image\warp_" + str(self.count) + ".jpg", warped)

        # Apply Sobel edge detection
        # sobel_binary = self.sobel_edge_detection(warped)

        hls_binary = self.hls_transform(warped)

        # white_part = self.get_white_part(warped)

        # combined_image = self.combine_image(hls_binary, sobel_binary)

        
        
        # Find lanes using sliding windows
        all_windows, out_img, detected_windows = self.sliding_windows(hls_binary)

        self.count += 1
        # cv2.imwrite("/home/ran/envs/CARLA_0.9.13/myProject/test_img/hls_" + str(self.count) + ".jpg", hls_binary * 255)
        # cv2.imwrite("D:\Apps\carla\CARLA_0.9.14\WindowsNoEditor\image\hls_" + str(self.count) + ".jpg", hls_binary *255)
        # cv2.imwrite("D:\Apps\carla\CARLA_0.9.14\WindowsNoEditor\image\out_" + str(self.count) + ".jpg", out_img)
        # cv2.imwrite("D:\Apps\carla\CARLA_0.9.14\WindowsNoEditor\image\sobel_" + str(self.count) + ".jpg", sobel_binary * 255)
        

        # Display results (optional, for debugging)
        # cv2.imshow("Lane Detection", hls_binary)
        # cv2.waitKey(1)
        # return None, None, None
        return all_windows, out_img, detected_windows

    def image_callback(self, data):
            self.lane_detect(data)




def main():
    detector = LaneDetector()


if __name__ == '__main__':
    main()