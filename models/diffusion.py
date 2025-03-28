import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np
from torch import nn
from .common import *
from zuko.utils import odeint
import rvo23d
import time
from multiprocessing import Pool

class VarianceSchedule(Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas



class PointwiseNet(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(point_dim, 128, context_dim+3),
            ConcatSquashLinear(128, 256, context_dim+3),
            ConcatSquashLinear(256, 512, context_dim+3),
            ConcatSquashLinear(512, 256, context_dim+3),
            ConcatSquashLinear(256, 128, context_dim+3),
            ConcatSquashLinear(128, point_dim, context_dim+3)
        ])

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out

class ZeroToOneTimeEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.register_buffer('freqs', torch.arange(1, dim // 2 + 1) * torch.pi)

    def forward(self, t):
        emb = self.freqs * t[..., None]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class CFM_Network(nn.Module):
    def forward(self, x, context, time):
        raise NotImplementedError()

class CFMNeuralNetwork(CFM_Network):

    def __init__(self, point_dim, context_dim, residual=False, time_embedding_size=8):
        super().__init__()
        self.time_embedding = ZeroToOneTimeEmbedding(time_embedding_size)
        hidden_size = context_dim + time_embedding_size
        self.act = F.relu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(point_dim, 128, hidden_size),
            ConcatSquashLinear(128, 256, hidden_size),
            ConcatSquashLinear(256, 512, hidden_size),
            ConcatSquashLinear(512, 256, hidden_size),
            ConcatSquashLinear(256, 128, hidden_size),
            ConcatSquashLinear(128, point_dim, hidden_size)
        ])


    def forward(self, x, context, time):
        batch_size = x.size(0)
        context = context.view(batch_size, 1, -1)  # (B, 1, F)
        time_emb = self.time_embedding(time).view(batch_size, 1, -1)  # (B, 1, time_embedding_size)
        ctx_emb = torch.cat([time_emb, context], dim=-1)  # (B, 1, F+3)
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)
        if self.residual:
            return x + out
        else:
            return out


class Orca():
    def __init__(self, timeStep=0.1, neighborDist=0.1, maxNeighbors=20, timeHorizon=1, radius=0.05, maxSpeed=1, velocity=(0, 0, 0), batch_size=1):
        self.timeStep = timeStep
        self.neighborDist = neighborDist
        self.maxNeighbors = maxNeighbors
        self.timeHorizon = timeHorizon
        self.radius = radius
        self.maxSpeed = maxSpeed
        self.velocity = velocity
        self.batch_size = batch_size


    def run(self, agents, velocity):
        batch_size, num_points, point_dim = agents.size()
        new_velocities = torch.zeros_like(velocity)
        sim = [rvo23d.PyRVOSimulator(self.timeStep, self.neighborDist, self.maxNeighbors, self.timeHorizon, self.radius, self.maxSpeed, self.velocity) for _ in range(batch_size)]
        for b in range(batch_size):
            agents_b = [sim[b].addAgent(tuple(position.tolist())) for position in agents[b, :, :]]
            for i in range(num_points):
                sim[b].setAgentPrefVelocity(agents_b[i], tuple(velocity[b][i].tolist()))
            
            sim[b].doStep()
            n_v = [sim[b].getAgentVelocity(agent) for agent in agents_b]
            new_velocities[b, :] = torch.tensor(n_v)
       
        del sim
        return new_velocities


class CFM(Module):

    def __init__(self, flow_model, args=None):
        super().__init__()
        self.flow_model = flow_model
        self.delta_t = 1
        self.args = args
        self.orca = Orca(radius=self.args.security_distance_value, batch_size=args.train_batch_size)

    def get_loss(self, x, code, t=None):
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
            context:  Shape latent, (B, F).
        """
        # t0 = time.time()
        # B, N, d = state_0.size()
        sigma_min = 1e-4
        t = torch.rand(x.shape[0], device=x.device)
        if self.args.prior_distribution == 'normal':
            noise = torch.randn_like(x)
        else:
            noise = torch.rand_like(x)*2 - 1

        x_t = (1 - (1 - sigma_min) * (1-t[:, None, None])) * noise + (1-t[:, None, None]) * x
        optimal_flow = x - (1 - sigma_min) * noise
        predicted_flow = self.flow_model(x_t, context=code, time=t)

        if self.args.orca_training:
            optimal_flow = self.orca.run(agents=x_t, velocity=optimal_flow)

        return (predicted_flow - optimal_flow).square().mean()


    def run_flow(self, code, x_T, t_0, T, num_points, device='cpu', ret_traj=False, num_steps=None):
        x_T = torch.randn([code.size(0), num_points, 3]).to(device)
        self.code = code
        def f(t: float, x):
            x_t = self.flow_model(x, context=self.code, time=torch.full(x.shape[:1], t).to(x.device))
            return x_t
        if ret_traj:
            trajectories = []
            for t in np.linspace(T, t_0, num_steps):
                if t == T:
                    x_t = x_T
                else:
                    x_t = odeint(f, x_T, T, t, phi=self.flow_model.parameters())
                trajectories.append(x_t)
            stacked_trajectories = torch.stack(trajectories, dim=0)
            return stacked_trajectories.transpose(0, 1)
        else:
            x_0_pred = odeint(f, x_T, T, t_0, phi=self.flow_model.parameters())
            return x_0_pred.unsqueeze(dim=1)


    def run_flow_progressively(self, code, t_0, T, num_points, device='cpu', ret_traj=False, num_steps=None):
        x_t = torch.randn([code.size(0), num_points, 3]).to(device)
        self.code = code
        def f(t: float, x):
            x_t = self.flow_model(x, context=self.code, time=torch.full(x.shape[:1], t).to(x.device))
            return x_t
        if ret_traj:
            trajectories = []
            times = np.linspace(T, t_0, num_steps)
            for k in range(len(times)-1):
                x_t = odeint(f, x_t, times[k], times[k-1], phi=self.flow_model.parameters())
                trajectories.append(x_t)
            stacked_trajectories = torch.stack(trajectories, dim=0)
            return stacked_trajectories.transpose(0, 1)
        else:
            x_0_pred = odeint(f, x_t, T, t_0, phi=self.flow_model.parameters())
            return x_0_pred.unsqueeze(dim=1)

    def run_flow_orca(self, code, x_T, t_0, T, num_points, device='cpu', ret_traj=False, num_steps=None):
        x_t = x_T

        times = np.linspace(T, t_0, num_steps)
        delta_t = 1/num_steps
        trajectories = []

        sim = rvo23d.PyRVOSimulator(delta_t, self.args.neighborDist, self.args.maxNeighbors, self.args.timeHorizon,
                                    self.args.radius, self.args.maxSpeed)
        agents_b = [sim.addAgent(tuple(position.tolist())) for position in x_t[0, :, :]]
        for t in times:
            flow_t = self.flow_model(x_t, context=code, time=torch.full(x_t.shape[:1], t).to(x_t.device))
            if self.args.orca_sampling:
                batch_size, num_points, point_dim = x_t.size()
                for i in range(num_points):
                    sim.setAgentPrefVelocity(agents_b[i], tuple(flow_t[0][i].tolist()))
                sim.doStep()
                x_t = torch.stack([torch.tensor(sim.getAgentPosition(agent)) for agent in agents_b]).unsqueeze(dim=0).to(device)
            else:
                x_t = x_t + flow_t*delta_t

            if ret_traj:
                trajectories.append(x_t)
        del sim
        if ret_traj:
            stacked_trajectories = torch.stack(trajectories, dim=0)
            return stacked_trajectories.transpose(0, 1)
        else:
            return x_t.unsqueeze(dim=1)