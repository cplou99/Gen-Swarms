import argparse
import json
from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.vae_gaussian import *
from models.vae_flow import *
from evaluation import *
import shutil

def normalize_point_clouds(pcs, mode, logger):
    if mode is None:
        logger.info('Will not normalize point clouds.')
        return pcs
    logger.info('Normalization mode: %s' % mode)
    for i in tqdm(range(pcs.size(0)), desc='Normalize'):
        for j in range(pcs.size(1)):
            pc = pcs[i][j]
            if mode == 'shape_unit':
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = pc.flatten().std().reshape(1, 1)
            elif mode == 'shape_bbox':
                pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
                pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
                shift = ((pc_min + pc_max) / 2).view(1, 3)
                scale = (pc_max - pc_min).max().reshape(1, 1) / 2
            pc = (pc - shift) / scale
            pcs[i][j] = pc
    return pcs

# Arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--ckpt', type=str, default='/home/cplou/PycharmProjects/Diffusion/flow/logs_gen/GEN_airplane2024_07_16__14_28_54/ckpt_700000.000000_300000.pt')
parser.add_argument('--ckpt', type=str, default='/home/pablo/Desktop/python/diffusion-master/flow/logs_final_gen/ShapeBbox/Airplane/ckpt_880000.000000_120000.pt')
# parser.add_argument('--ckpt', type=str, default='/home/cplou/PycharmProjects/Diffusion/flow/logs_gen/GEN_FLOW_airplane2024_07_22__12_25_28/ckpt_820000.000000_180000.pt')

# parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--save_dir', type=str, default='/results_diff')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='/home/pablo/Desktop/python/diffusion-master/flow/data/shapenet.hdf5')
parser.add_argument('--batch_size', type=int, default=64)
# Sampling
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--seed', type=int, default=9988)
parser.add_argument('--orca_training', type=eval, default=False, choices=[True, False]) #new line
parser.add_argument('--security_net', type=eval, default=False, choices=[True, False]) #new line
parser.add_argument('--security_distance_value', type=float, default=0.02) #new line
parser.add_argument('--orca_sampling', type=eval, default=True, choices=[True, False]) #new line

parser.add_argument('--neighborDist', type=float, default=0.1) #new line
parser.add_argument('--maxNeighbors', type=int, default=50) #new line
parser.add_argument('--timeHorizon', type=float, default=0.05) #new line
parser.add_argument('--radius', type=float, default=0.03) #new line
parser.add_argument('--maxSpeed', type=float, default=6) #new line
parser.add_argument('--num_steps', type=int, default=100) #new line

args = parser.parse_args()

save_dir = "final_shapes/flow_orca_50"
all_pcs = torch.from_numpy(np.load(save_dir + '/all_pcs.npy'))
ref_pcs = torch.from_numpy(np.load(save_dir + '/refs.npy'))

all_pcs = all_pcs
# Compute metrics
with (torch.no_grad()):
    results_vel = {}#compute_smoothness_metrics(all_pcs.to(args.device))
    metrics_vel = {}
    metrics_col = {}
    metrics_recons = {}

    # Calculate the metrics for each list in results_vel
    for key, data in results_vel.items():
        data_array = np.array(data)
        metrics_vel[key] = {
            'mean': float(np.mean(data_array)),
            'median': float(np.median(data_array)),
            'std': float(np.std(data_array)),
            'min': float(np.min(data_array)),
            'max': float(np.max(data_array))
        }
    col_path = os.path.join(save_dir, 'col_metrics.json')
    vel_path = os.path.join(save_dir, 'vel_metrics.json')
    recons_path = os.path.join(save_dir, 'recons_metrics.json')
    full_vel_path = os.path.join(save_dir, 'full_vel_metrics.json')
    full_col_path = os.path.join(save_dir, 'full_col_metrics.json')

    print(metrics_vel)
    with open(full_vel_path, 'w') as file:
        for key in results_vel:
            results_vel[key] = [f"{index}:{float(x)}" for index, x in enumerate(results_vel[key])]
        json.dump(results_vel, file, indent=4)

    threshold = 2 * args.radius * 100 / 3
    results_cols = compute_collisions_metrics(all_pcs.to(args.device), threshold)

    # Calculate the metrics for each list in results_vel
    for key, data in results_cols.items():
        if not key == 'all_cols':
            data_array = np.array(data)
            metrics_col[key] = {
                'mean': float(np.mean(data_array)),
                'median': float(np.median(data_array)),
                'std': float(np.std(data_array)),
                'min': float(np.min(data_array)),
                'max': float(np.max(data_array))
            }

    with open(full_col_path, 'w') as file:
        for key in results_cols:
            if not key == 'all_cols':
                results_cols[key] = [f"{index}:{float(x)}" for index, x in enumerate(results_cols[key])]
        results_cols['threshold'] = threshold
        json.dump(results_cols, file, indent=4)

    finals = all_pcs[:, -1, :, :]
    finals = finals.to(torch.float32)

    results_recons = compute_recons_metrics(finals.to(args.device), ref_pcs.to(args.device), args.batch_size)

    # results_recons = compute_recons_metrics(finals.to(args.device), ref_pcs.to(args.device), args.batch_size)

    # Calculate the metrics for each list in results_vel
    for key, data in results_recons.items():
        data_array = np.array(data.cpu())
        metrics_recons[key] = {
            'mean': float(np.mean(data_array)),
            'median': float(np.median(data_array)),
            'std': float(np.std(data_array)),
            'min': float(np.min(data_array)),
            'max': float(np.max(data_array))
        }

    # Save the metrics to a JSON file
    with open(col_path, 'w') as file:
        json.dump(metrics_col, file)
    with open(recons_path, 'w') as file:
        json.dump(metrics_recons, file)
    with open(vel_path, 'w') as file:
        json.dump(metrics_vel, file)


    print("Results vel: ", results_vel)
    print("Results cols: ", results_cols)
    print("Results recons: ", results_recons)


    # Calculate the metrics

    last_dir = "/home/pablo/Desktop/python/diffusion-master/flow/last_results/"
    if os.path.exists(last_dir):
        shutil.rmtree(last_dir)
    shutil.copytree(save_dir, last_dir)

    #results_recons = {k:v.item() for k, v in results_recons.items()}
   # results_vel = {k:v.item() for k, v in results_vel.items()}
   # results_cols = {k:v.item() for k, v in results_cols.items()}

   # jsd = jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
    #results['jsd'] = jsd
