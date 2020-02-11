import cv2
import sys
import numpy as np

def compute_num_inliers(homography_matrix, sampled_keypoints, threshold):
    len_kp = len(sampled_keypoints)
    inliers = []
    for i in range(0, len_kp):
        # Get x and y of src and dest keypoints at current index of array of sampled keypoints
        x, y = sampled_keypoints[i][0].pt
        x_prime, y_prime = sampled_keypoints[i][1].pt
        # Add 1.0 for 'z' dimension of point x,y and multiply with homography matrix (3x3)
        src_x, src_y, src_z = np.matmul(homography_matrix, (x,y,1.0))
        # Divide out z coord to get standard coordinate
        src_x_std = src_x/src_z
        src_y_std = src_y/src_z
        # Get euclidean distance between the keypoint from image 2 and the keypoint from image 1 after transformation
        euclidean_distance = np.linalg.norm(np.array([x_prime, y_prime], np.float32) - np.array([src_x_std, src_y_std], np.float32))
        # If the distance is within the bound of the threshold, add the keypoint from image 1 to the list of inliers
        if euclidean_distance < threshold:
            inliers.append([sampled_keypoints[i][0], sampled_keypoints[i][1]])
    return inliers

def find_homography(sampled_keypoints):
    len_sample = len(sampled_keypoints)
    if len_sample == 0:
        return None
    h_matrix = []
    for i in range(0, len_sample):
        # Get all of the src keypoints
        x, y = sampled_keypoints[i][0].pt
        # Get all of the dest keypoints
        x_prime, y_prime = sampled_keypoints[i][1].pt
        # Calculate two rows of the matrix
        to_add = [x, y, 1, 0, 0, 0, -x_prime*x, -x_prime*y, -x_prime]
        to_add_prime = [0, 0, 0, x, y, 1, -y_prime*x, -y_prime*y, -y_prime]
        # Append the two rows of matrix to homography matrix
        h_matrix.append(to_add)
        h_matrix.append(to_add_prime)
    # Use numpy to solve for homography given calculated matrix.
    u, s, vh = np.linalg.svd(np.array(h_matrix, np.float32))
    # Get last line of resulting unit array, then reshape it to 3x3. Divide out w (last element in last line) to get final homography matrix.
    homography = vh[-1,:].reshape(3,3) / vh[-1,-1]
    return homography

def ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000):
    '''
    Apply RANSAC algorithm to find a homography transformation matrix that align 2 sets of feature points, transform the first set of feature point to the second (e.g. warp image 1 to image 2)
    :param list_pairs_matched_keypoints: has the format as a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]],....]
    :param threshold_ratio_inliers: threshold on the ratio of inliers over the total number of samples, accept the estimated homography if ratio is higher than the threshold
    :param threshold_reprojtion_error: threshold of reprojection error (measured as euclidean distance, in pixels) to determine whether a sample is inlier or outlier
    :param max_num_trial: the maximum number of trials to do take sample and do testing to find the best homography matrix
    :return best_H: the best found homography matrix
    '''
    best_H = None
    largest_num_inliers = []
    num_samples = 4
    len_lpmk = len(list_pairs_matched_keypoints)

    for iterations in range(0, max_num_trial):
        sampled_keypoints = []
        # Randomly select 4 pairs of keypoints
        sample_indices = np.random.choice(len_lpmk, size=num_samples, replace=False)
        for i in range(0, num_samples):
           sampled_keypoints.append(list_pairs_matched_keypoints[sample_indices[i]]) 
        # Calculate homography matrix based on 4 pairs of matched keypoints
        possible_H = find_homography(sampled_keypoints)
        # Evaluate the current fit:
        inliers = compute_num_inliers(possible_H, list_pairs_matched_keypoints, threshold_reprojtion_error)
        num_inliers = len(inliers)
        if num_inliers/num_samples > threshold_ratio_inliers:
            # If current number of inliers is more than previous, then this is the best matrix so far
            if num_inliers > len(largest_num_inliers):
                largest_num_inliers = inliers
                best_H = possible_H
    best_H = find_homography(largest_num_inliers)
    return best_H

# Detects sift features by first converting to grayscale and then using opencv detectAndCompute function
# Returns 2 lists of keypoints, and 2 lists of their corresponding descriptors
def detect_sift_features(img_1, img_2):
    im1 = img_1.copy()
    im2 = img_2.copy()
    gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    sift1 = cv2.xfeatures2d.SIFT_create()
    sift2 = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift1.detectAndCompute(gray,None)
    kp2, des2 = sift2.detectAndCompute(gray2,None)
    cv2.drawKeypoints(gray,kp1,im1)
    cv2.drawKeypoints(gray2,kp2,im2)
    return kp1, des1, kp2, des2

# Matches the closest keypoints in the image, assuming the ratio of the closest and 2nd closest distance < 0.7
# Returns the index in keypoint array where the closest descriptor was found, or an invalid index if the ratio
# was >= 0.7
def find_matches(descriptor_to_match, descriptor_match, ratio_robustness):
    kp_m_len = len(descriptor_match)
    dist1 = 999
    dist2 = 999
    index_of_matched_keypoint = 0
    # Compare distance between the descriptor to match and the current descriptor in dest array.
    for i in range(0, kp_m_len):
        curr_dist = np.linalg.norm(descriptor_to_match - descriptor_match[i])
        # If this is the smallest distance so far, the second closest distance now equals closest distance.
        # Closest distance is updated to the current distance.
        if curr_dist < dist1:
            dist2 = dist1
            dist1 = curr_dist
            # Save the current index if it is the closest so far.
            index_of_matched_keypoint = i
        elif curr_dist < dist2:
            dist2 = curr_dist
    # Compare distance of closest feature to the distance of second closest feature.
    if dist1/dist2 < ratio_robustness:
        return index_of_matched_keypoint
    # Return an invalid index if the ratio d1/d2 is >= the robustness ratio. This feature will be ignored.
    return -1

def ex_extract_and_match_feature(img_1, img_2, ratio_robustness=0.7):
    '''
    1/ extract SIFT feature from image 1 and image 2,
    2/ use a bruteforce search to find pairs of matched features: for each feature point in img_1, find its best matched feature point in img_2
    3/ apply ratio test to select the set of robust matched points
    :param img_1: input image 1
    :param img_2: input image 2
    :param ratio_robustness: ratio for the robustness test
    :return list_pairs_matched_keypoints: has the format as list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]]]
    '''
    # ==============================
    # ===== 1/ extract features from input image 1 and image 2
    # ==============================
    keypoints1, descriptors1, keypoints2, descriptors2 = detect_sift_features(img_1, img_2)
    # ==============================
    # ===== 2/ use bruteforce search to find a list of pairs of matched feature points
    # ==============================
    list_pairs_matched_keypoints = []
    len1 = len(keypoints1)
    len2 = len(keypoints2)
    for i in range(0, len1):
        # Pass in each keypoint's descriptor, and compare it with all of the other image's keypoint descriptors.
        match = find_matches(descriptors1[i], descriptors2, ratio_robustness)
        # If the match was within the robustness ratio, append to a list of matched keypoints pairs.
        if match >= 0:
            list_pairs_matched_keypoints.append([keypoints1[i], keypoints2[match]])
    return list_pairs_matched_keypoints

# Blends two images by averaging areas where both contain values that are > 0. (ie: if the pixel in both images
# to be blended is not black, average them and copy to new image.)
def blend_images(canvas_img1, canvas_img2):
    canvas_height = canvas_img1.shape[0]
    canvas_width = canvas_img1.shape[1]
    combined_images = np.zeros_like(canvas_img2, dtype = np.float32)
    # Where pixels overlap, average them
    for i in range(0, canvas_height):
        for j in range(0, canvas_width):
            if canvas_img1[i][j].any() > 0 and canvas_img2[i][j].any() > 0:
                for k in range(0,3):
                   combined_images[i][j][k] = (canvas_img2[i][j][k] + canvas_img1[i][j][k])/2.0
            elif canvas_img1[i][j].any() > 0 or canvas_img2[i][j].any() > 0:
                if canvas_img1[i][j].any() > 0:
                    for k in range(0,3):
                        combined_images[i][j][k] = canvas_img1[i][j][k]
                if canvas_img2[i][j].any() > 0:
                    for k in range(0,3):
                       combined_images[i][j][k] = canvas_img2[i][j][k]
    return combined_images

# Copies destination image to center of canvas that is 3*h X 3*w
def copy_image2_to_canvas(img_2, canvas_height, canvas_width, canvas_img2, bounds):
    height = img_2.shape[0]
    width = img_2.shape[1]
    for i in range(0, canvas_height):
        for j in range(0, canvas_width):
            x = i - height;
            y = j - width;
            if (x >= 0 and x < height) and (y >= 0 and y < width):
                for color in range(0,3):
                    canvas_img2[i][j][color] = img_2[x][y][color]
                # Calculate largest and smallest x,y values. These will be used to create the bounding box
                # to crop the final image.
                if j < bounds[0]:
                    bounds[0] = j
                if j > bounds[1]:
                    bounds[1] = j
                if i < bounds[2]:
                    bounds[2] = i
                if i > bounds[3]:
                    bounds[3] = i
    return canvas_img2

# Applies inverse homography to source image and interpolates 
def apply_homography_and_interpolate(img_1, canvas_img1, canvas_width, canvas_height, bounds, inverse_H):
    height = img_1.shape[0]
    width = img_1.shape[1]
    weight = 0
    if width == 250:
        weight = 75
    else:
        weight = 50
    # Apply inverse homography matrix to image 1 and copy to canvas of image1
    for i in range(0, canvas_width):
        for j in range(0, canvas_height):
            # Calculate current coordinate in range -h, 2h and -w, 2w.
            x_coord = i - width
            y_coord = j - height
            # Apply inverse homography to image.
            x_s_h, y_s_h, z_s_h = np.matmul(inverse_H, (y_coord, x_coord, 1.0))

            # Convert to Cartesian coordinates.
            x = y_s_h/z_s_h 
            y = x_s_h/z_s_h
            # Check to see if coordinates are out of range.
            if(x >= 0 and x < height-1 and y >= 0 and y < width-1):
                # a,b are set to the decimal value of floating point numbers. x,y are set to the integer part.
                a = x-int(x)
                b = y-int(y)
                x = int(x)
                y = int(y)
                #Calculate weights for bilinear interpolation.
                weight1 = (1-a)*(1-b)
                weight2 = a*(1-b)
                weight3 = b*a
                weight4 = (1-a)*b
                # Apply bilinear interpolation to each channel of image.
                for color in range(0, 3):
                    canvas_img1[i-weight][j+weight][color] = weight1 * img_1[x][y][color] + weight2 * img_1[x+1][y][color] + weight3 * img_1[x+1][y+1][color] + weight4 * img_1[x][y+1][color]
                # Calculate largest and smallest x,y values. These will be used to create the bounding box
                # to crop the final image.
                if j+weight < bounds[0]:
                    bounds[0] = j+weight #smallest_x = j+weight
                if j+weight > bounds[1]:
                    bounds[1] = j+weight #largest_x = j+weight
                if i-weight < bounds[2]:
                    bounds[2] = i-weight #smallest_y = i-weight
                if i-weight > bounds[3]:
                    bounds[3] = i-weight #largest_y = i-weight
    return canvas_img1

# Applies bounds found in process of copying and warping images to canvas to crop image to minimum area.
def apply_best_bounding_box(bounds, img_panorama, combined_images):
    x1 = int(bounds[0])+1 #smallest x
    y1 = int(bounds[2])+1 #smallest y
    x2 = int(bounds[1])+1 #largest x
    y2 = int(bounds[3])+1 #largest y
	
    h = abs(x2 - x1)
    w = abs(y2 - y1)
    img_panorama = np.zeros_like(combined_images, dtype = np.float32)
    img_panorama.resize(w,h,3)
    for i in range(y1, y2):
        for j in range(x1, x2):
            if combined_images[i][j].any() > 0:
                for k in range(0,3):
                    img_panorama[i-y1][j-x1][k] = combined_images[i][j][k]

    return img_panorama

def ex_warp_blend_crop_image(img_1,H_1,img_2):
    '''
    1/ warp image img_1 using the homography H_1 to align it with image img_2 (using backward warping and bilinear resampling)
    2/ stitch image img_1 to image img_2 and apply average blending to blend the 2 images into a single panorama image
    3/ find the best bounding box for the resulting stitched image
    :param img_1:
    :param H_1:
    :param img_2:
    :return img_panorama: resulting panorama image
    '''
    img_panorama = None
    # =====  use a backward warping algorithm to warp the source
    # 1/ to do so, we first create the inverse transform; 2/ use bilinear interpolation for resampling

    # Array to hold the bounds of the warped image: this will be used to crop the final image.
    bounds = [999,0,999,0]
    height = img_1.shape[0]
    width = img_1.shape[1]
    canvas_height = 3*height
    canvas_width = 3*width
    # Canvas for source image: will be 3w * 3h in size.
    canvas_img1 = np.zeros_like(img_1, dtype=np.float32)
    canvas_img1.resize(canvas_height, canvas_width,3)
    # Canvas for destination image: will be 3w * 3h in size.
    canvas_img2 = np.zeros_like(canvas_img1, dtype = np.float32) 
    # Inverse homography matrix.
    inverse_H = np.linalg.inv(H_1)

    # Copy img_2 to center of temporary canvas for image 2
    canvas_img2 = copy_image2_to_canvas(img_2, canvas_height, canvas_width, canvas_img2, bounds)

    # Apply inverse homography matrix to image 1 and copy to canvas of image1
    canvas_img1 = apply_homography_and_interpolate(img_1, canvas_img1, canvas_width, canvas_height, bounds, inverse_H)

    # ===== blend images: average blending
    combined_images = blend_images(canvas_img1, canvas_img2)

    # ===== find the best bounding box for the resulting stitched image so that it will contain all pixels from 2 original images

    img_panorama = apply_best_bounding_box(bounds, img_panorama, combined_images)

    return img_panorama

def stitch_images(img_1, img_2):
    '''
    :param img_1: input image 1. We warp this image to align and stich it to the image 2
    :param img_2: is the reference image. We will not warp this image
    :return img_panorama: the resulting stiched image
    '''
    print('==============================')
    print('===== stitch two images to generate one panorama image')
    print('==============================')

    # ===== extract and match features from image 1 and image 2
    list_pairs_matched_keypoints = ex_extract_and_match_feature(img_1=img_1, img_2=img_2, ratio_robustness=0.7)

    # ===== use RANSAC algorithm to find homography to warp image 1 to align it to image 2
    H_1 = ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000)

    # ===== warp image 1, blend it with image 2 using average blending to produce the resulting panorama image
    img_panorama = ex_warp_blend_crop_image(img_1=img_1,H_1=H_1, img_2=img_2)


    return img_panorama

if __name__ == "__main__":
    print('==================================================')
    print('PSU CS 410/510, Winter 2019, HW2: image stitching')
    print('==================================================')

    path_file_image_1 = sys.argv[1]
    path_file_image_2 = sys.argv[2]
    path_file_image_result = sys.argv[3]


    # ===== read 2 input images
    img_1 = cv2.imread(path_file_image_1)
    img_2 = cv2.imread(path_file_image_2)

    # ===== create a panorama image by stitch image 1 to image 2
    img_panorama = stitch_images(img_1=img_1, img_2=img_2)

    # ===== save panorama image
    cv2.imwrite(filename=path_file_image_result, img=(img_panorama).clip(0.0, 255.0).astype(np.uint8))

