import os
import time
import math
import argparse
import torch
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.vae_gaussian import *
from models.vae_flow import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from evaluation import *
import shutil
import torch.distributions as dist
import json

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
parser.add_argument('--ckpt', type=str, default='./logs_gen/gen-swarms_airplane.pt')
parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--batch_size', type=int, default=1)

# Sampling
parser.add_argument('--num_gen_samples', type=int, default=5)
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--seed', type=int, default=998)
parser.add_argument('--orca_training', type=eval, default=False, choices=[True, False]) #new line
parser.add_argument('--security_net', type=eval, default=False, choices=[True, False]) #new line
parser.add_argument('--security_distance_value', type=float, default=0.01) #new line
parser.add_argument('--orca_sampling', type=eval, default=True, choices=[True, False]) #new line
parser.add_argument('--real_scale', type=float, default=100) #new line

parser.add_argument('--neighborDist', type=float, default=0.5) #new line
parser.add_argument('--maxNeighbors', type=int, default=50) #new line
parser.add_argument('--timeHorizon', type=float, default=0.05) #new line
parser.add_argument('--radius', type=float, default=0.03) #new line
parser.add_argument('--maxSpeed', type=float, default=6) #new line
parser.add_argument('--num_steps', type=int, default=50) #new line
parser.add_argument('--transition', type=eval, default=False, choices=[True, False]) #new line
parser.add_argument('--prior_distribution', type=str, default='normal', choices=['normal', 'uniform']) #new line
args = parser.parse_args()


# Logging
save_dir = os.path.join(args.save_dir, 'GEN_Ours_%s_%d' % ('_'.join(args.categories), int(time.time())) )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Checkpoint
ckpt = torch.load(args.ckpt)
seed_all(args.seed)

# Datasets and loaders
logger.info('Loading datasets...')
test_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='test',
    scale_mode=ckpt['args'].scale_mode,
)
test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=0)

# Model
ckpt['args'].orca_training = args.orca_training
ckpt['args'].security_net = args.security_net
ckpt['args'].security_distance_value = args.security_distance_value
ckpt['args'].orca_sampling = args.orca_sampling
ckpt['args'].neighborDist = args.neighborDist
ckpt['args'].maxNeighbors = args.maxNeighbors
ckpt['args'].timeHorizon = args.timeHorizon
ckpt['args'].radius = args.radius
ckpt['args'].maxSpeed = args.maxSpeed
ckpt['args'].prior_distribution = args.prior_distribution

logger.info('Loading model...')
if ckpt['args'].model == 'gaussian':
    model = GaussianVAE(ckpt['args']).to(args.device)
elif ckpt['args'].model == 'flow':
    model = FlowVAE(ckpt['args']).to(args.device)
logger.info(repr(model))
model.load_state_dict(ckpt['state_dict'])

# Reference Point Clouds
ref_pcs = []
for i, data in enumerate(test_dset):
    ref_pcs.append(data['pointcloud'].unsqueeze(0))
ref_pcs = torch.cat(ref_pcs, dim=0)

# Generate Point Clouds
gen_pcs, all_pcs = [], []
for i in tqdm(range(0, math.ceil(len(test_dset) / args.batch_size)), 'Generate'):
    with torch.no_grad():
        z = torch.randn([args.batch_size, ckpt['args'].latent_dim]).to(args.device)
        if args.transition:
            if i == 0:
                if args.prior_distribution == 'normal':
                    x_t = torch.randn([args.batch_size, args.sample_num_points, 3]).to(args.device)
                else:
                    x_t = torch.rand([args.batch_size, args.sample_num_points, 3]).to(args.device) * 2 - 1
                x_T = x_t
                trajectories = torch.stack([x_t], dim=1)
            else:
                x_t = x[:, -1, :, :]
                if args.prior_distribution == 'normal':
                    goal = torch.randn([args.batch_size, args.sample_num_points, 3]).to(args.device)
                else:
                    goal = torch.rand([args.batch_size, args.sample_num_points, 3]).to(args.device) * 2 - 1

                delta_t = 1 / args.num_steps
                trajectories = [x_t]
                if args.orca_sampling:
                    sim = rvo23d.PyRVOSimulator(delta_t, args.neighborDist, args.maxNeighbors, args.timeHorizon,
                                                args.radius, args.maxSpeed)
                    agents_b = [sim.addAgent(tuple(position.tolist())) for position in x_t[0, :, :]]
                j = 0
                while torch.norm(goal - x_t) > 5:
                    flow_t = goal - x_t
                    print("Dist at step", j, torch.norm(goal - x_t))
                    if args.orca_sampling:
                        batch_size, num_points, point_dim = flow_t.size()
                        for k in range(num_points):
                            sim.setAgentPrefVelocity(agents_b[k], tuple(flow_t[0][k].tolist()))
                        sim.doStep()
                        x_t = torch.stack([torch.tensor(sim.getAgentPosition(agent)) for agent in agents_b]).unsqueeze(
                            dim=0).to(args.device)
                    else:
                        flow_t = flow_t / torch.norm(flow_t, dim=-1, keepdim=True)
                        x_t = x_t + flow_t * delta_t * args.maxSpeed
                    j += 1
                    trajectories.append(x_t)
                x_T = x_t
                trajectories = torch.stack(trajectories, dim=1)
        else:
            if args.prior_distribution == 'normal':
                x_T = torch.randn([args.batch_size, args.sample_num_points, 3]).to(args.device)
            else:
                x_T = torch.rand([args.batch_size, args.sample_num_points, 3]).to(args.device) * 2 - 1

        x = model.sample(z, x_T, args.sample_num_points, device=args.device, ret_traj=True, num_steps=args.num_steps)
        gen_pcs.append(x[:, -1, :, :].detach().cpu())
        if args.transition:
            x = torch.cat([trajectories, x, x[:, -1, :, :].repeat(1, 20, 1, 1)], dim=1)
        all_pcs.append(x.detach().cpu())
    
    if i == args.num_gen_samples:
        print("We have reached the number of samples to be generated and will break. WARNING: metrics must be computed from the whole dataset (args.num_gen_samples=None). This is just to get some visualization examples.")
        break

gen_pcs = torch.cat(gen_pcs, dim=0)[:len(test_dset)]
if args.transition:
    all_pcs = torch.cat(all_pcs, dim=1)
else:
    all_pcs = torch.cat(all_pcs, dim=0)[:len(test_dset)]


if args.normalize is not None:
    gen_pcs = normalize_point_clouds(gen_pcs.unsqueeze(dim=1), mode=args.normalize, logger=logger)
    all_pcs = normalize_point_clouds(all_pcs, mode=args.normalize, logger=logger)

# gen_pcs = gen_pcs.squeeze(dim=1)
print("Rescaling to real scale")
all_pcs = all_pcs * args.real_scale 
gen_pcs = gen_pcs * args.real_scale 
ref_pcs = ref_pcs * args.real_scale 


# Save
logger.info('Saving point clouds...')
np.save(os.path.join(save_dir, 'out.npy'), gen_pcs.numpy())
np.save(os.path.join(save_dir, 'all_pcs.npy'), all_pcs.numpy())



# Compute metrics
with (torch.no_grad()):
    results_vel = compute_smoothness_metrics(all_pcs.to(args.device))
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

    threshold = 2 * args.radius * args.real_scale / 3
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
    last_dir = "./last_results/"
    if os.path.exists(last_dir):
        shutil.rmtree(last_dir)
    shutil.copytree(save_dir, last_dir)


