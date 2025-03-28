import torch
from torch.nn import Module

from .common import *
from .encoders import *
from .diffusion import *


class GaussianVAE(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(args.latent_dim)
        if self.args.security_net:
            security_net = SecurityNet(point_dim=3, n_points=args.sample_num_points, residual=args.residual)
        else:
            security_net = None
        self.diffusion = DiffusionPoint(
            FlowNeuralNetwork(point_dim=3, context_dim=args.latent_dim, residual=args.residual), args=self.args,
            security_net=security_net)
        
    def get_loss(self, x, writer=None, it=None, kl_weight=1.0, wandb=None):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        batch_size, _, _ = x.size()
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        log_pz = standard_normal_logprob(z).sum(dim=1)  # (B, ), Independence assumption
        entropy = gaussian_entropy(logvar=z_sigma)      # (B, )
        loss_prior = (- log_pz - entropy).mean()

        loss_recons = self.diffusion.get_loss(x, z)

        loss = kl_weight * loss_prior + loss_recons

        if writer is not None:
            writer.add_scalar('train/loss_entropy', -entropy.mean(), it)
            writer.add_scalar('train/loss_prior', -log_pz.mean(), it)
            writer.add_scalar('train/loss_recons', loss_recons, it)

        if wandb is not None:
            train_dict = {'loss_entropy': -entropy.mean().item(), 'loss_prior': -log_pz.mean().item(), 'loss_recons': loss_recons.item()}
            wandb.log(train_dict, step=it)

        return loss

    def sample(self, z, x_T, num_points, device, truncate_std=None, ret_traj=False, num_steps=None):
        """
        Args:
            z:  Input latent, normal random samples with mean=0 std=1, (B, F)
        """
        if truncate_std is not None:
            z = truncated_normal_(z, mean=0, std=1, trunc_std=truncate_std)
        samples = self.diffusion.run_flow(z, x_T, 0, 1, num_points, device=device, ret_traj=ret_traj, num_steps=num_steps)
        return samples
