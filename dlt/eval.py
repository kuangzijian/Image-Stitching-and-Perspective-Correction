import os
import numpy as np
import argparse
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--srcdir", help = "path to the images directory")
    ap.add_argument("-n", "--gtdir", help = "path to npy groundtruth directory")
    ap.add_argument("--norm", action="store_true")
    args = ap.parse_args(['-i', 'images', '-n', 'gt'])

    error = 0
    files = os.listdir(args.srcdir)
    for image_path in files:
        image = cv2.imread(os.path.join(args.srcdir, image_path))
        npy_file = os.path.join(args.gtdir, image_path.split('/')[-1].replace('png', 'npy'))
        gt = np.load(npy_file, allow_pickle=True).item()
        points = gt['points']
        homography_gt = gt['homography']

        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        target_points = compute_target_points(pts)

        # TODO: Call your DLT methods according to norm flag
        # homography = dlt(points)
        # error += np.linalg.norm(homography.flatten() - homography_gt.flatten())**2

        # visualize the result
        mask = np.zeros((image.shape[0], image.shape[1]))
        cv2.fillConvexPoly(mask, pts, 1)
        mask = mask.astype(np.bool)
        cropped = np.zeros_like(image)
        cropped[mask] = image[mask]

        cv2.polylines(image, [pts], True, (0, 255, 0), 2)
        warped = cv2.warpPerspective(cropped, homography_gt, (cropped.shape[:2][1], cropped.shape[:2][0]))
        concatenated = np.hstack((image, warped)).astype(np.uint8)
        cv2.imshow('results', concatenated)
        cv2.waitKey(1500)

    print('Total Mean Squared Error: ', error/len(files))

def dlt(points):
    homography = []
    return homography

def compute_target_points(src_points):
  # Order points and compute src points in the image
  rect = order_points(src_points)
  (tl, tr, br, bl) = rect
  # compute the width of the new image, which will be the
  # maximum distance between bottom-right and bottom-left
  # x-coordiates or the top-right and top-left x-coordinates
  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))
  # compute the height of the new image, which will be the
  # maximum distance between the top-right and bottom-right
  # y-coordinates or the top-left and bottom-left y-coordinates
  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))
  dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
  return dst

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = -1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = -1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

if __name__ == "__main__":
    main()
