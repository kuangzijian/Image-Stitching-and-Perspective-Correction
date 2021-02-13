import cv2
import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image_dir", help="path to the image")
    ap.add_argument("-r", "--results_dir", help="path to the visualization result")
    #args = ap.parse_args(['-i', '../vanishing_pt/carla.png', '-r', '../results/'])
    args = ap.parse_args()

    img = cv2.imread(args.image_dir)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 120, 150, 10)
    img1 = img.copy()
    img2 = img.copy()

    # Use OpenCV for line detection
    lines = cv2.HoughLinesP(edges, rho=1, theta=1*np.pi/180, threshold=100, minLineLength=50, maxLineGap=150)
    N = lines.shape[0]
    for i in range(N):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]
        cv2.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Assume that the vanishing point is intersection point of most of the lines
    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i + 1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))
    most_point = most_frequent(intersections)

    # # Visualize the detected lines
    cv2.imshow('line detection', img1)
    cv2.imwrite(args.results_dir+'/line_detection.png', img1)

    # Visualize the vanishing point
    cv2.circle(img2, (most_point[0][0], most_point[0][1]), radius=6, color=(0, 0, 255), thickness=3)
    cv2.imshow('vanishing point', img2)
    cv2.imwrite(args.results_dir + '/vanishing_point.png', img2)
    print("You can also review the generated visualization results in the 'results' folder. Press any key to exit.")
    cv2.waitKey()


def intersection(line1, line2):
    line1 = [[line1[0], line1[1]], [line1[2], line1[3]]]
    line2 = [[line2[0], line2[1]], [line2[2], line2[3]]]
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        x = 0
        y = 0
    else:
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div

    return [[int(x), int(y)]]


def most_frequent(List):
    counter = 0
    point = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            point = i

    return point

if __name__ == "__main__":
    main()