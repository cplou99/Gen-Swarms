import torch
from torch.nn import Module

from .common import *
from .encoders import *
from .diffusion import *
from .flow import *


class FlowVAE(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(args.latent_dim)
        self.flow = build_latent_flow(args)
        self.diffusion = CFM(CFMNeuralNetwork(point_dim=3, context_dim=args.latent_dim, residual=args.residual), args=self.args)


    def get_loss(self, x, kl_weight, writer=None, it=None, wandb=None):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        batch_size, _, _ = x.size()
        # print(x.size())
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        
        # H[Q(z|X)]
        entropy = gaussian_entropy(logvar=z_sigma)      # (B, )

        # P(z), Prior probability, parameterized by the flow: z -> w.
        w, delta_log_pw = self.flow(z, torch.zeros([batch_size, 1]).to(z), reverse=False)
        log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(dim=1, keepdim=True)   # (B, 1)
        log_pz = log_pw - delta_log_pw.view(batch_size, 1)  # (B, 1)

        # Negative ELBO of P(X|z)
        neg_elbo = self.diffusion.get_loss(x, z)

        # Loss
        loss_entropy = -entropy.mean()
        loss_prior = -log_pz.mean()
        loss_recons = neg_elbo
        loss = kl_weight*(loss_entropy + loss_prior) + neg_elbo

        if writer is not None:
            writer.add_scalar('train/loss_entropy', loss_entropy, it)
            writer.add_scalar('train/loss_prior', loss_prior, it)
            writer.add_scalar('train/loss_recons', loss_recons, it)
            writer.add_scalar('train/z_mean', z_mu.mean(), it)
            writer.add_scalar('train/z_mag', z_mu.abs().max(), it)
            writer.add_scalar('train/z_var', (0.5*z_sigma).exp().mean(), it)

        if wandb is not None:
            train_dict = {'loss_entropy': loss_entropy.item(), 'loss_prior': loss_prior.item(), 'loss_recons': loss_recons.item(), 'z_mean': z_mu.mean().item(), 'z_mag': z_mu.abs().max().item(), 'z_var': (0.5*z_sigma).exp().mean().item()}
            wandb.log(train_dict, step=it)
        return loss

    def sample(self, w, x_T, num_points, device, truncate_std=None, ret_traj=False, num_steps=None):
        batch_size, _ = w.size()
        if truncate_std is not None:
            w = truncated_normal_(w, mean=0, std=1, trunc_std=truncate_std)
        # Reverse: z <- w.
        z = self.flow(w, reverse=True).view(batch_size, -1)
        # samples = self.diffusion.run_flow(z, 0, 1, num_points, device=device, ret_traj=ret_traj, num_steps=num_steps)
        samples = self.diffusion.run_flow_orca(z, x_T,  0, 1, num_points, device=device, ret_traj=ret_traj, num_steps=num_steps)
        return samples
