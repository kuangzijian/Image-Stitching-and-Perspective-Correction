import numpy as np
from skimage.transform import ProjectiveTransform
from dlt.dlt import dlt
def ransac(data, threshold_distance, threshold_inliers=0, max_trials=100):
    random_state = np.random.mtrand._rand
    num_samples = len(data[0])

    for num_trials in range(max_trials):
        # Sample Random 4 point pairs correspondences (at least 3 non collinear)
        spl_idxs = random_state.choice(num_samples, 4, replace=False)
        samples = [d[spl_idxs] for d in data]

        # Estimate homography H using DLT algorithm
        H = dlt(samples[0], samples[1])

        # Reproject All points using H and compute reprojection error
        src_pts = np.insert(data[0], 2, 1, axis=1).T
        dst_pts = np.insert(data[1], 2, 1, axis=1).T
        projected_pts = np.dot(H, src_pts)
        error = np.sqrt(np.sum(np.square(dst_pts - (projected_pts/projected_pts[-1])), axis=0))


        count_inliers = np.count_nonzero(error < threshold_distance)
        if count_inliers/num_samples > threshold_inliers:
            sample_model_inliers = error < threshold_distance
            src_inliers = src_pts[:, np.argwhere(error < threshold_distance).flatten()][:-1].T
            dst_inliers = dst_pts[:, np.argwhere(error < threshold_distance).flatten()][:-1].T

            # if number of inliers/number of total samples is over threshold then re-estimate H using inliers and stop
            H = dlt(src_inliers, dst_inliers)
            break

    return H, sample_model_inliers