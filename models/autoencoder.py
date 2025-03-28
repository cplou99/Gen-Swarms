import torch
from torch.nn import Module

from .encoders import *
from .diffusion import *


class AutoEncoder(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(zdim=args.latent_dim)
        if args.security_net:
            security_net = SecurityNet(point_dim=3, n_points=args.sample_num_points, residual=args.residual)
        else:
            security_net = None
        self.diffusion = DiffusionPoint(FlowNeuralNetwork(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
                                        security_net=security_net,
                                        args=self.args)

    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        """
        code, _ = self.encoder(x)
        return code

    def decode(self, code, num_points, device, ret_traj=False, num_steps=None):
        return self.diffusion.run_flow(code, 0, 1, num_points, device=device, ret_traj=ret_traj, num_steps=num_steps)

    def get_loss(self, x_0):
        batch_size, _, point_dim = x_0.size()
        # pos = state[:, :, :point_dim//2]
        code = self.encode(x_0)
        loss = self.diffusion.get_loss(x_0, code)
        return loss
