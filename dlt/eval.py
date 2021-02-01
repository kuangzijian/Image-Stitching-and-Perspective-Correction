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

        # TODO: Call your DLT methods according to norm flag
        # homography = dlt(points)

        #error += np.linalg.norm(homography.flatten() - homography_gt.flatten())**2

    print('Total Mean Squared Error: ', error/len(files))

if __name__ == "__main__":
    main()
