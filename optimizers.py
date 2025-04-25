import torch, einops
from functools import partial

def orthogonalize_via_svd(matrix):
    """
    Orthogonalize a matrix using SVD decomposition.
    
    For a matrix M with SVD decomposition M = U·S·V^T, 
    this returns U·V^T which is the closest orthogonal matrix to M.
    """
    # Handle both regular and "fat" matrices (where width > height)
    is_fat = matrix.size(-2) < matrix.size(-1)
    if is_fat: matrix = einops.rearrange(matrix, '... i j -> ... j i')
        
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    result = U @ Vh
    
    if is_fat: result = einops.rearrange(result, '... i j -> ... j i')    
    return result

def orthogonalize_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.

    Taken from: https://github.com/KellerJordan/Muon/blob/master/muon.py
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class MomentumOrth(torch.optim.Optimizer):

    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, momentum_after_orth=False,
                 newtonschulz=False, ns_steps=5, orthogonalize=True, log_stats=False):

        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
            momentum_after_orth=momentum_after_orth, newtonschulz=newtonschulz, ns_steps=ns_steps
        )
        
        super().__init__([{'params': list(params)}], defaults)
        self.orthogonalize = orthogonalize
        self.log_stats = log_stats
        print('Logging opt stats:', log_stats)
    
    @torch.no_grad()
    def step(self):
        """Perform a single optimization step."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue

                # Initialize buffers if needed
                state = self.state[p]

                # Apply weight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])

                # Get gradient
                update = p.grad
                if self.log_stats: old_gradient = update.clone()

                if not self.orthogonalize:
                    update = self._apply_momentum(update, group['momentum'], group['nesterov'], state)
                    aspect_ratio_scaling = 1.0

                elif group['momentum_after_orth']:
                    # Orthogonalize first, then apply momentum
                    update, aspect_ratio_scaling = self._orth_update(update, state)
                    update = self._apply_momentum(update, group['momentum'], group['nesterov'], state)

                else:
                    # Muon-style
                    update = self._apply_momentum(update, group['momentum'], group['nesterov'], state)
                    update, aspect_ratio_scaling = self._orth_update(update, state)
                
                p.add_(update, alpha = - group['lr'] * aspect_ratio_scaling)

                if self.log_stats: self._log_stats(old_gradient, update, state, 'overall')
                
    
    def _orth_update(self, update, state):

        orthogonalize = partial(orthogonalize_via_newtonschulz5, steps = self.defaults['ns_steps']) if self.defaults['newtonschulz'] else orthogonalize_via_svd

        if self.log_stats: old_update = update.clone()

        # Handle different parameter shapes
        if update.ndim == 2:
            # Matrix case - directly orthogonalize
            output_dim, input_dim = update.shape
            update = orthogonalize(update)
            aspect_ratio_scaling = max(1, output_dim / input_dim) ** 0.5
            
        elif update.ndim == 4:
            # Convolutional filter case
            out_channels, in_channels, k_height, k_width = update.shape
            
            # Flatten last three dimensions, orthogonalize, and restore shape
            update = einops.rearrange(update, 'o i h w -> o (i h w)')
            update = orthogonalize(update)
            update = einops.rearrange(update, 'o (i h w) -> o i h w', i = in_channels, h = k_height, w = k_width)
            
            aspect_ratio_scaling = max(1, out_channels / (in_channels * k_height * k_width)) ** 0.5
            
        elif update.ndim == 3:
            # 3D tensor case - reshape, orthogonalize, reshape back
            dim_a, dim_b, dim_c = update.shape
            
            # Treat as (dim_a) × (dim_b × dim_c)
            update = einops.rearrange(update, 'a b c -> a (b c)')
            update = orthogonalize(update)
            update = einops.rearrange(update, 'a (b c) -> a b c', b = dim_b, c = dim_c)
            
            aspect_ratio_scaling = max(1, dim_a / (dim_b * dim_c)) ** 0.5
            
        else:
            aspect_ratio_scaling = 1.0

        if self.log_stats: self._log_stats(old_update, update, state, 'orth')
        
        return update, aspect_ratio_scaling

    def _apply_momentum(self, update, momentum, nesterov, state):

        if self.log_stats: old_update = update.clone() 

        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(update)
        momentum_buffer = state['momentum_buffer']
                
        momentum_buffer.lerp_(update, 1 - momentum)
        if nesterov:
            update = update.lerp(momentum_buffer, momentum)
        else:
            update = momentum_buffer

        if self.log_stats: self._log_stats(old_update, update, state, 'momentum')
        return update

    def _log_stats(self, old_update, new_update, state, label):
        """
        Log the statistics of the optimizer's state.
        """
        state[f'stat-{label}-grad-lF-diff'] = (old_update - new_update).norm().item()
        cos_sim = torch.nn.functional.cosine_similarity(old_update.flatten(), new_update.flatten(), dim=0)
        state[f'stat-{label}-grad-cos-sim'] = cos_sim.item()
        norm_ratio = torch.norm(new_update) / torch.norm(old_update)
        state[f'stat-{label}-grad-norm-ratio'] = norm_ratio.item()