import numpy as np

def dlt(src_pts, target_pts):
    # Construct A Matrix from pairs a and b
    A = []
    for i in range(0, len(src_pts)):
        ax, ay = src_pts[i][0], src_pts[i][1]
        bx, by = target_pts[i][0], target_pts[i][1]
        A.append([-ax, -ay, -1, 0, 0, 0, bx*ax, bx*ay, bx])
        A.append([0, 0, 0, -ax, -ay, -1, by*ax, by*ay, by])

    # Compute SVD for A
    A = np.asarray(A)
    U, S, V = np.linalg.svd(A)

    # The solution is the last column of V (9 x 1) Vector
    L = V[-1, :]

    # Divide by last element as we estimate the homography up to a scale
    L = L/V[-1, -1]
    H = L.reshape(3, 3)

    return H