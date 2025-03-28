"""
From https://github.com/stevenygd/PointFlow/tree/master/metrics
"""
import torch
import numpy as np
import warnings
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from tqdm.auto import tqdm
from collections import defaultdict


_EMD_NOT_IMPL_WARNED = False
def emd_approx(sample, ref):
    global _EMD_NOT_IMPL_WARNED
    emd = torch.zeros([sample.size(0)]).to(sample)
    if not _EMD_NOT_IMPL_WARNED:
        _EMD_NOT_IMPL_WARNED = True
        print('\n\n[WARNING]')
        print('  * EMD is not implemented due to GPU compatability issue.')
        print('  * We will set all EMD to zero by default.')
        print('  * You may implement your own EMD in the function `emd_approx` in ./evaluation/evaluation_metrics.py')
        print('\n')
    return emd


def our_emd(sample, ref):
    d = earth_mover_distance(ref, sample, transpose=False)  # p1: B x N1 x 3, p2: B x N2 x 3
    return d


# Borrow from https://github.com/ThibaultGROUEIX/AtlasNet
def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]


def EMD_CD(sample_pcs, ref_pcs, batch_size, reduced=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    cd_lst = []
    emd_lst = []
    iterator = range(0, N_sample, batch_size)

    for b_start in tqdm(iterator, desc='EMD-CD'):
        b_end = min(N_sample, b_start + batch_size)
        sample_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]

        dl, dr = distChamfer(sample_batch, ref_batch)
        cd_lst.append(dl.mean(dim=1) + dr.mean(dim=1))

        emd_batch = emd_approx(sample_batch, ref_batch)
        emd_lst.append(emd_batch)

    if reduced:
        cd = torch.cat(cd_lst).mean()
        emd = torch.cat(emd_lst).mean()
    else:
        cd = torch.cat(cd_lst)
        emd = torch.cat(emd_lst)

    results = {
        'MMD-CD': cd,
        'MMD-EMD': emd,
    }
    return results


def _pairwise_EMD_CD_(sample_pcs, ref_pcs, batch_size, verbose=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []
    iterator = range(N_sample)
    if verbose:
        iterator = tqdm(iterator, desc='Pairwise EMD-CD')
    for sample_b_start in iterator:
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        emd_lst = []
        sub_iterator = range(0, N_ref, batch_size)
        # if verbose:
        #     sub_iterator = tqdm(sub_iterator, leave=False)
        for ref_b_start in sub_iterator:
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            point_dim = ref_batch.size(2)
            sample_batch_exp = sample_batch.view(1, -1, point_dim).expand(
                batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()

            dl, dr = distChamfer(sample_batch_exp, ref_batch)
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))

            emd_batch = emd_approx(sample_batch_exp, ref_batch)
            emd_lst.append(emd_batch.view(1, -1))

        cd_lst = torch.cat(cd_lst, dim=1)
        emd_lst = torch.cat(emd_lst, dim=1)
        all_cd.append(cd_lst)
        all_emd.append(emd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
    all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref

    return all_cd, all_emd


def _compute_cols_metrics(all_trajs, threshold, verbose=True):
    N_sample = all_trajs.shape[0]

    all_cols_finals = []
    all_cols_traj = []
    all_cols_init = []
    all_cols_string = []

    iterator = range(N_sample)
    if verbose:
        iterator = tqdm(iterator, desc='Cols Metrics')
    for sample_b_start in iterator:
        sample_batch = all_trajs[sample_b_start]

        sample_batch_exp = sample_batch.unsqueeze(0)
        sample_batch_exp = sample_batch_exp.contiguous()

        #cols_final_batch = collisions_one(sample_batch_exp[0, sample_batch_exp.shape[1] - 1, :, :].cpu(), threshold) # taking the last step
        #cols_init_batch = collisions_one(sample_batch_exp[0, 0, :, :].cpu(), threshold)
        cols_traj_batch = collisions_traj(sample_batch_exp[0, :, :, :].cpu(), threshold)

        # Convert each number to a string
        string_numbers = map(str, cols_traj_batch)
        # Join them with commas
        comma_separated_string = ','.join(string_numbers)

        #all_cols_finals.append(cols_traj_batch[sample_batch_exp.shape[1] - 1])
        all_cols_finals.append(cols_traj_batch[sample_batch_exp.shape[1] - 2])
        all_cols_traj.append(np.mean(cols_traj_batch))
        all_cols_init.append(cols_traj_batch[0])
        all_cols_string.append(comma_separated_string)

    results = {
        'cols_init': all_cols_init,
        'cols_finals': all_cols_finals,
        'mean': all_cols_traj,
        'all_cols': all_cols_string
    }
    return results
def _compute_smoothness_metrics(all_trajs, verbose=True):
    N_sample = all_trajs.shape[0]
    num_steps = all_trajs.shape[1]
    delta_t = 1 / 1#num_steps

    all_vel_dir = []
    all_acc = []
    all_jerk = []
    all_distances = []
    iterator = range(N_sample)

    if verbose:
        iterator = tqdm(iterator, desc='Smoothness Metrics')
    for sample_b_start in iterator:
        sample_batch = all_trajs[sample_b_start]

        sample_batch_exp = sample_batch.unsqueeze(0)
        sample_batch_exp = sample_batch_exp.contiguous()

        distances = calculate_mean_distance(sample_batch_exp.cpu())
        mean_acceleration_norm_batch, _, _, mean_jerk_norm_batch, _, _ = acc_jerk(sample_batch_exp.cpu(),    delta_t)
        vel_dir_batch = vel_direction(sample_batch_exp.cpu(), delta_t)

        #vel_vector_batch = vel_rms_deviation(sample_batch_exp.cpu(), delta_t)
        #acc_batch = acc_module(sample_batch_exp.cpu(), 1)

        all_vel_dir.append(vel_dir_batch)
        all_acc.append(mean_acceleration_norm_batch)
        all_jerk.append(mean_jerk_norm_batch)
        all_distances.append(distances)

    results = {
        'vel_dir': all_vel_dir,
        'acc': all_acc,
        'jerk': all_jerk,
        'distance': all_distances
    }
    return results

# Adapted from https://github.com/xuqiantong/
# GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat([
        torch.cat((Mxx, Mxy), 1),
        torch.cat((Mxy.transpose(0, 1), Myy), 1)], 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(
        k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s


def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }


def lgan_mmd_cov_match(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }, min_idx.view(-1)
def compute_smoothness_metrics(all_trajs):
    M_s_s = _compute_smoothness_metrics(all_trajs)
    return M_s_s

def compute_collisions_metrics(all_trajs, threshold):
    M_s_s = _compute_cols_metrics(all_trajs, threshold)
    return M_s_s

def compute_recons_metrics(sample_pcs, ref_pcs, batch_size):
    results = {}

    # print("Pairwise EMD CD")
    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(ref_pcs, sample_pcs, batch_size)

    ## CD COV
    res_cd = lgan_mmd_cov(M_rs_cd.t())
    results.update({
        "%s-CD" % k: v for k, v in res_cd.items()
    })
    #
    # ## EMD COV
    # res_emd = lgan_mmd_cov(M_rs_emd.t())
    # results.update({
    #     "%s-EMD" % k: v for k, v in res_emd.items()
    # })

    M_rr_cd, M_rr_emd = _pairwise_EMD_CD_(ref_pcs, ref_pcs, batch_size)
    M_ss_cd, M_ss_emd = _pairwise_EMD_CD_(sample_pcs, sample_pcs, batch_size)

    # 1-NN results
    ## CD
    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
    results.update({
        "1-NN-CD-%s" % k: v for k, v in one_nn_cd_res.items() if 'acc' in k
    })
    # EMD
    # one_nn_emd_res = knn(M_rr_emd, M_rs_emd, M_ss_emd, 1, sqrt=False)
    # results.update({
    #     "1-NN-EMD-%s" % k: v for k, v in one_nn_emd_res.items() if 'acc' in k
    # })

    for k, v in results.items():
        print('[%s] %.8f' % (k, v.item()))
    return results


#######################################################
# JSD : from https://github.com/optas/latent_3d_points
#######################################################
def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with
    resolution^3 cells, that is placed in the unit-cube. If clip_sphere it True
    it drops the "corner" cells that lie outside the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing

def calculate_distances(points):
    num_points = points.shape[0]
    distances = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i + 1, num_points):
            distance = np.linalg.norm(points[i] - points[j])
            distances[i, j] = distance
            distances[j, i] = distance
    return distances

def collisions_one(pc, threshold):
    differences = pc.unsqueeze(1) - pc.unsqueeze(0).detach()
    dist = torch.norm(differences, dim=-1)
    dist[dist == 0] = np.inf
    min_distance = dist.min(dim=1)[0]
    #min_distance_all = dist.min()
    penalty_mask_sdv = (min_distance < threshold)
    points_colliding = torch.sum(penalty_mask_sdv)
    # percentage_sdv = (points_colliding.item() / pc.shape[0]) * 100
    # results = {
    #     'points_colliding': points_colliding,
    #     'percentage_coll': percentage_sdv,
    #     'min_d': dist.min(),
    #     'max_d': dist.max()
    # }
    return points_colliding.item()

def collisions_traj(traj, threshold):
    collisions = []
    # Iterate over batches and time steps
    for step_idx in range(0,traj.shape[0]-1):
        collisions.append(collisions_one(traj[step_idx], threshold))
    return collisions

def vel_rms_deviation(all_pcs, delta_t):
    # Step 1: Calculate the velocities (differences between consecutive time steps)
    velocities = np.diff(all_pcs, axis=1) / delta_t  # Shape will be (b, 200, 256, 3)
    # Step 2: Calculate the mean velocity along the time axis
    mean_velocity = np.mean(velocities, axis=1, keepdims=True)  # Shape will be (b, 1, 256, 3)
    # Step 3: Calculate the deviation of each velocity from the mean velocity
    deviation_from_mean = velocities - mean_velocity  # Shape will be (b, 200, 256, 3)
    # Step 4: Square the deviations
    squared_deviation = np.square(deviation_from_mean)  # Shape will be (b, 200, 256, 3)
    # Step 5: Calculate the mean of the squared deviations along the time axis
    mean_squared_deviation = np.mean(squared_deviation, axis=1)  # Shape will be (b, 256, 3)
    # Step 6: Compute the RMS deviation
    rms_deviation = np.sqrt(mean_squared_deviation)  # Shape will be (b, 256, 3)
    # Step 7: Calculate the average RMS deviation across all batches and points
    average_rms_deviation = np.mean(rms_deviation)  # Scalar value

    return average_rms_deviation
def calculate_mean_distance(trajectory):
    # Remove the singleton dimension
    tensor = np.squeeze(trajectory)  # Shape becomes (101, 2048, 3)

    # Compute the distances between consecutive steps
    distances = np.linalg.norm(np.diff(tensor, axis=0), axis=2)  # Shape (100, 2048)

    # Sum the distances for each point
    total_distances = np.sum(distances, axis=0)  # Shape (2048,)

    # total_distances now contains the distance each point has traversed
    print(total_distances)

    return np.mean(total_distances)



def acc_jerk(trajectory, delta_t):
    """
    Compute smoothness metrics of a trajectory by analyzing acceleration and jerk.

    Parameters:
    - trajectory: numpy array of shape (n_samples, n_steps, n_points, n_dimensions)
    - delta_t: time interval between consecutive steps

    Returns:
    - mean_acceleration_norm: Mean of the acceleration norms across all steps and points
    - variance_acceleration_norm: Variance of the acceleration norms across all steps and points
    - rms_acceleration_norm: Root Mean Square of the acceleration norms across all steps and points
    - mean_jerk_norm: Mean of the jerk norms across all steps and points
    - variance_jerk_norm: Variance of the jerk norms across all steps and points
    - rms_jerk_norm: Root Mean Square of the jerk norms across all steps and points
    """

    # Compute velocities: shape (n_samples, n_steps-1, n_points, n_dimensions)
    velocities = np.diff(trajectory, axis=1) / delta_t

    # Compute accelerations: shape (n_samples, n_steps-2, n_points, n_dimensions)
    accelerations = np.diff(velocities, axis=1) / delta_t

    # Compute the norm of accelerations: shape (n_samples, n_steps-2, n_points)
    acceleration_norms = np.linalg.norm(accelerations, axis=-1)

    # Compute jerk: shape (n_samples, n_steps-3, n_points, n_dimensions)
    jerks = np.diff(accelerations, axis=1) / delta_t
    jerk_norms = np.linalg.norm(jerks, axis=-1)

    # Compute smoothness metrics for acceleration
    mean_acceleration_norm = np.mean(acceleration_norms)
    variance_acceleration_norm = np.var(acceleration_norms)
    rms_acceleration_norm = np.sqrt(np.mean(acceleration_norms ** 2))

    # Compute smoothness metrics for jerk
    mean_jerk_norm = np.mean(jerk_norms)
    variance_jerk_norm = np.var(jerk_norms)
    rms_jerk_norm = np.sqrt(np.mean(jerk_norms ** 2))

    return (mean_acceleration_norm, variance_acceleration_norm, rms_acceleration_norm,
            mean_jerk_norm, variance_jerk_norm, rms_jerk_norm)


def vel_direction(all_pcs, delta_t):
    """
    Compute the variation in the direction of velocities.

    Parameters:
    - all_pcs: numpy array of shape (n_samples, n_steps, n_points, n_dimensions)
    - delta_t: time interval between consecutive steps

    Returns:
    - mean_direction_variation: Mean of the standard deviations of angles between consecutive velocities
    """

    # Step 1: Calculate the velocities (differences between consecutive time steps)
    velocities = np.diff(all_pcs, axis=1) / delta_t  # Shape: (n_samples, n_steps-1, n_points, n_dimensions)
    # Step 2: Compute norms of velocities
    norms = np.linalg.norm(velocities, axis=-1, keepdims=True)  # Shape: (n_samples, n_steps-1, n_points, 1)
    # Step 3: Normalize velocities
    normalized_velocities = velocities / norms  # Shape: (n_samples, n_steps-1, n_points, n_dimensions)
    # Step 4: Compute dot products between consecutive normalized velocities
    dot_products = np.einsum('ijkl,ijkm->ijk', normalized_velocities[:, :-1],
                             normalized_velocities[:, 1:])  # Shape: (n_samples, n_steps-2, n_points)
    # Ensure dot_products are within the valid range for arccos due to floating point precision
    dot_products = np.clip(dot_products, -1.0, 1.0)
    # Convert dot products to angles (in radians)
    angles = np.arccos(dot_products)  # Shape: (n_samples, n_steps-2, n_points)
    # Compute the standard deviation of the angles across all time steps to measure variation in direction
    direction_variation = np.std(angles, axis=1)  # Shape: (n_samples, n_points)
    # Return the mean of the standard deviations
    mean_direction_variation = np.mean(direction_variation)  # Scalar value

    return mean_direction_variation

def jsd_between_point_cloud_sets(
        sample_pcs, ref_pcs, resolution=28):
    """Computes the JSD between two sets of point-clouds,
       as introduced in the paper
    ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    """
    in_unit_sphere = True
    sample_grid_var = entropy_of_occupancy_grid(
        sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(
        ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


def entropy_of_occupancy_grid(
        pclouds, grid_resolution, in_sphere=False, verbose=False):
    """Given a collection of point-clouds, estimate the entropy of
    the random variables corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    """
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in tqdm(pclouds, desc='JSD'):
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)  # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def _jsdiv(P, Q):
    """another way of computing JSD"""

    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))


if __name__ == '__main__':
    a = torch.randn([16, 2048, 3]).cuda()
    b = torch.randn([16, 2048, 3]).cuda()
    print(EMD_CD(a, b, batch_size=8))
    