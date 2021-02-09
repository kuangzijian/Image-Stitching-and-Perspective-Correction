import numpy as np

def dltnorm(src_pts, target_pts):
    # Compute a similarity trasformation T and T_prime to normalize src_pts and target_pts
    normalized_src_pts, T = normalization(src_pts)
    normalized_target_pts, T_prime = normalization(target_pts)

    # Construct A Matrix from pairs a and b
    A = []
    for i in range(0, len(normalized_src_pts)):
        ax, ay = normalized_src_pts[i][0], normalized_src_pts[i][1]
        bx, by = normalized_target_pts[i][0], normalized_target_pts[i][1]
        A.append([-ax, -ay, -1, 0, 0, 0, bx*ax, bx*ay, bx])
        A.append([0, 0, 0, -ax, -ay, -1, by*ax, by*ay, by])

    # Compute SVD for A
    A = np.asarray(A)
    U, S, V = np.linalg.svd(A)

    # The solution is the last column of V (9 x 1) Vector
    L = V[-1, :]

    # Divide by last element as we estimate the homography up to a scale
    L = L/V[-1, -1]
    H_tilde = L.reshape(3, 3)

    # Denormalization: denormalize the homography back
    H = np.dot(np.dot(np.linalg.pinv(T_prime), H_tilde), T)
    H = H/H[-1, -1]

    return H

def normalization(pts):
    N = len(pts)
    mean = np.mean(pts, 0)
    s = np.linalg.norm((pts-mean), axis=1).sum() / (N * np.sqrt(2))

    # Compute a similarity transformation T, moves original points to
    # new set of points, such that the new centroid is the origin,
    # and the average distance from origin is square root of 2
    T = np.array([[s, 0, mean[0]],
                  [0, s, mean[1]],
                  [0, 0, 1]])
    T = np.linalg.inv(T)
    pts = np.dot(T, np.concatenate((pts.T, np.ones((1, pts.shape[0])))))
    pts = pts[0:2].T
    return pts, T
