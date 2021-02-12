import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import match_descriptors
from skimage.feature import plot_matches
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from ransac import ransac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--first_image_dir", help="path to the first image")
    ap.add_argument("-s", "--second_image_dir", help="path to the second image")
    ap.add_argument("-r", "--results_dir", help="path to the visualization result")
    ap.add_argument("--lmeds", action="store_true")
    args = ap.parse_args(['-f', '../overdetermined/office/office-00.png',
                          '-s', '../overdetermined/office/office-01.png',
                          '-r', '../results',
                          '--lmeds'])
    #args = ap.parse_args()

    #I0 = cv2.imread("../overdetermined/fishbowl/fishbowl-00.png")
    I0 = cv2.imread(args.first_image_dir)
    pano0 = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
    #I1 = cv2.imread("../overdetermined/fishbowl/fishbowl-01.png")
    I1 = cv2.imread(args.second_image_dir)
    pano1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)

    # Using Scale Invariant Feature Transform (SIFT) to detect image features
    sift = cv2.SIFT_create(400)

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

    # Find best matches using Ransac
    homograhpy, inliers01 = ransac((src, dst), threshold_distance=0.8, threshold_inliers=0.3, max_trials=500)

    # Best match subset for pano0 -> pano1
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plot_matches(ax, I0, I1, keypoints0, keypoints1, matches[inliers01])
    ax.axis('off')
    fig.suptitle('Best matching after RANSAC (or LMedS)', fontsize=20)
    fig.savefig(args.results_dir+'/ransac_matching.png', dpi=fig.dpi)
    cv2.imshow('ransac matching', cv2.imread(args.results_dir+'/ransac_matching.png'))
    cv2.waitKey(10)

    # Image warping and stitching
    xh = np.linalg.inv(homograhpy)
    ds = np.dot(xh, np.array([pano1.shape[1], pano1.shape[0], 1]))
    ds = ds / ds[-1]
    print("final ds=>", ds)
    f1 = np.dot(xh, np.array([0, 0, 1]))
    f1 = f1 / f1[-1]
    xh[0][-1] += abs(f1[0])
    xh[1][-1] += abs(f1[1])
    ds = np.dot(xh, np.array([pano1.shape[1], pano1.shape[0], 1]))
    offsety = abs(int(f1[1]))
    offsetx = abs(int(f1[0]))
    dsize = (int(ds[0]) + offsetx, int(ds[1]) + offsety)
    print("image dsize =>", dsize)
    tmp = cv2.warpPerspective(a, xh, dsize)

    print(pano0.shape[1] + offsetx)
    t = tmp[offsety:pano0.shape[0] + offsety, offsetx:pano0.shape[1] + offsetx].shape
    pano0 = pano0[:, 0:t[1]]
    tmp[offsety:pano0.shape[0] + offsety, offsetx:pano0.shape[1] + offsetx] = pano0

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plt.imshow(tmp, cmap="gray")
    fig.suptitle('Image Stitching Result', fontsize=20)
    fig.savefig(args.results_dir+'/stitching_result.png', dpi=fig.dpi)
    cv2.imshow('stitching result', cv2.imread(args.results_dir+'/stitching_result.png'))
    print("You can also review the generated visualization results in the 'results' folder. Press any key to exit.")
    cv2.waitKey()


if __name__ == "__main__":
    main()
