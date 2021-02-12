import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import match_descriptors
from skimage.feature import plot_matches
from skimage.transform import SimilarityTransform
from skimage.transform import ProjectiveTransform
from skimage.transform import warp
from ransac import ransac
from lmeds import lmeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--first_image_dir", help="path to the first image")
    ap.add_argument("-s", "--second_image_dir", help="path to the second image")
    ap.add_argument("-r", "--results_dir", help="path to the visualization result")
    ap.add_argument("--lmeds", action="store_true")
    # args = ap.parse_args(['-f', '../overdetermined/office/office-00.png',
    #                       '-s', '../overdetermined/office/office-01.png',
    #                       '-r', '../results'])
    args = ap.parse_args()

    I0 = cv2.imread(args.first_image_dir)
    pano0 = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
    I1 = cv2.imread(args.second_image_dir)
    pano1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)

    # Using Scale Invariant Feature Transform (SIFT) to detect image features
    sift = cv2.SIFT_create(400)
    print('1. Using Scale Invariant Feature Transform (SIFT) to detect image features')

    # Detect keypoints in pano0 using OpenCV SIFT detect function
    kp0, des0 = sift.detectAndCompute(pano0, None)

    # Detect keypoints in pano1 using OpenCV SIFT detect function
    kp1, des1 = sift.detectAndCompute(pano1, None)

    # Visualize the detected and matched features
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Detected and matched features', fontsize=20)
    plt.subplot(121)
    plt.imshow(cv2.drawKeypoints(pano0, kp0, None), cmap="gray")
    plt.title("Image 0 keypoints")
    plt.subplot(122)
    plt.imshow(cv2.drawKeypoints(pano1, kp1, None), cmap="gray")
    plt.title("Image 1 keypoints")
    fig.savefig(args.results_dir+'/keypoints.png', dpi=fig.dpi)
    cv2.imshow('keypoints', cv2.imread(args.results_dir+'/keypoints.png'))
    cv2.waitKey(10)

    # Match descriptors between images
    matches = match_descriptors(des0, des1, cross_check=True)
    print('2. Implement a simple feature matching and visualize the detected and matched features')

    # Restore the openCV style keypoints into a 2d array type keypoints
    keypoints0 = []
    a = 0
    for i in kp0:
        keypoints0.append(list(kp0[a].pt)[::-1])
        a += 1
    keypoints0 = np.array(keypoints0)

    keypoints1 = []
    b = 0
    for j in kp1:
        keypoints1.append(list(kp1[b].pt)[::-1])
        b += 1
    keypoints1 = np.array(keypoints1)

    # Best match subset for pano0 -> pano1
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plot_matches(ax, I0, I1, keypoints0, keypoints1, matches)
    ax.axis('off')
    fig.suptitle('Initial matching', fontsize=20)
    fig.savefig(args.results_dir+'/initial_matching.png', dpi=fig.dpi)
    cv2.imshow('initial matching', cv2.imread(args.results_dir+'/initial_matching.png'))
    cv2.waitKey(10)

    # Select keypoints from
    #   * source (image to be registered): pano1
    #   * target (reference image): pano0
    src = keypoints1[matches[:, 1]][:, ::-1]
    dst = keypoints0[matches[:, 0]][:, ::-1]

    # Find best matches using Ransac or LMedS
    if(args.lmeds):
        homography, inliers01 = lmeds((src, dst), threshold_distance=0.8, threshold_inliers=0.3, max_trials=500)
        print('3. Using LMedS to find the best matching')
        title = 'Best matching after LMedS'
    else:
        homography, inliers01 = ransac((src, dst), threshold_distance=0.8, threshold_inliers=0.3, max_trials=500)
        print('3. Using RANSAC to find the best matching')
        title = 'Best matching after RANSAC'

    # Best match subset for pano0 -> pano1
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plot_matches(ax, I0, I1, keypoints0, keypoints1, matches[inliers01])
    ax.axis('off')
    fig.suptitle(title, fontsize=20)
    fig.savefig(args.results_dir+'/ransac_matching.png', dpi=fig.dpi)
    cv2.imshow('ransac matching', cv2.imread(args.results_dir+'/ransac_matching.png'))
    cv2.waitKey(10)

    # Image warping and stitching
    print('4. Perform Image warping and stitching')
    result = stitching(homography, pano0, I0, I1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plt.imshow(result, cmap="gray")
    fig.suptitle('Image Stitching Result', fontsize=20)
    fig.savefig(args.results_dir+'/stitching_result.png', dpi=fig.dpi)
    cv2.imshow('stitching result', cv2.imread(args.results_dir+'/stitching_result.png'))
    print("You can also review the generated visualization results in the 'results' folder. Press any key to exit.")
    cv2.waitKey()

def stitching(homography, pano0, I0, I1):
    # Shape registration target
    r, c = pano0.shape[:2]

    # Note that transformations take coordinates in (x, y) format,
    # not (row, column), in order to be consistent with most literature
    corners = np.array([[0, 0, 1],
                        [0, r, 1],
                        [c, 0, 1],
                        [c, r, 1]])

    # Warp the image corners to their new positions
    warped_corners01 = np.dot(homography, corners.T)
    warped_corners01 = warped_corners01[:2, :].T

    # Find the extents of both the reference image and the warped
    # target image
    all_corners = np.vstack((warped_corners01, corners[:, :2]))

    # The overally output shape will be max - min
    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)
    output_shape = (corner_max - corner_min)

    # Ensure integer shape with np.ceil and dtype conversion
    output_shape = np.ceil(output_shape[::-1]).astype(int)

    # This in-plane offset is the only necessary transformation for the middle image
    offset1 = SimilarityTransform(translation=-corner_min)
    tform = ProjectiveTransform(homography)

    # Warp pano1 to pano0 using 3rd order interpolation
    transform01 = (tform + offset1).inverse
    I1_warped = warp(I1, transform01, order=3,
                     output_shape=output_shape, cval=-1)

    I1_mask = (I1_warped != -1)  # Mask == 1 inside image
    I1_warped[~I1_mask] = 0  # Return background values to 0

    # Translate pano0 into place
    I0_warped = warp(I0, offset1.inverse, order=3,
                     output_shape=output_shape, cval=-1)

    I0_mask = (I0_warped != -1)  # Mask == 1 inside image
    I0_warped[~I0_mask] = 0  # Return background values to 0

    # Add the images together. This could create dtype overflows!
    # We know they are are floating point images after warping, so it's OK.
    merged = (I0_warped + I1_warped)

    # Track the overlap by adding the masks together
    overlap = (I0_mask * 1.0 +  # Multiply by 1.0 for bool -> float conversion
               I1_mask)

    # Normalize through division by `overlap` - but ensure the minimum is 1
    normalized = merged / np.maximum(overlap, 1)
    return normalized

if __name__ == "__main__":
    main()

