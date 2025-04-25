from functools import partial
import numpy as np
import torch, einops

class DiffLogger:
    def __init__(self, modules):
        self.before, self.after, self.diff = {}, {}, {}
        self._state = None
        self.modules = modules
        self.hooks = []

        self.value_fns = {
            'inputs': lambda m, i, o: i[0],
            'outputs': lambda m, i, o: o
        }

        for name, module in modules.items():
            for label, fn in self.value_fns.items():
                hook = module.register_forward_hook(
                    partial(self._log_forward, module_id=name, label=label, value_fn=fn)
                )
                self.hooks.append(hook)

    def set_state(self, state):
        assert state in ['before', 'after']
        self._state = state

    def _log_forward(self, module, input, output, module_id, label, value_fn):
        key = (module_id, label)
        val = value_fn(module, input, output).detach().clone()

        if self._state == 'before':
            self.before[key] = val
        elif self._state == 'after':
            self.after[key] = val
            try:
                self.diff[key] = val - self.before[key]
            except Exception as e:
                print(f'[DiffLogger] Failed to compute diff for {key}: {e}')
                self.diff[key] = None

    def log_weights(self):
        for name, module in self.modules.items():
            if hasattr(module, 'weight'):
                key = (name, 'weights')
                val = module.weight.data.detach().clone()
                if self._state == 'before':
                    self.before[key] = val
                elif self._state == 'after':
                    self.after[key] = val
                    try:
                        self.diff[key] = val - self.before[key]
                    except Exception as e:
                        print(f'[DiffLogger] Failed to compute weight diff for {key}: {e}')
                        self.diff[key] = None

    def clear(self):
        self.before.clear()
        self.after.clear()
        self.diff.clear()
        self._state = None

    def close(self):
        for h in self.hooks:
            h.remove()

    def __del__(self):
        self.close()



def spectral_norm(A):
    try:
        return torch.linalg.matrix_norm(A, ord = 2).item()
    except Exception as e:
        print(f'[spectral_norm] Failed to compute spectral norm: {e}')
        return None

def condition_number(A):
    # Ratio of the largest singular value to the smallest
    try:
        U, S, Vh = torch.linalg.svd(A, full_matrices = False)
        if S[-1] < 1e-6:
            return float('inf')
        return (S[0] / S[-1]).item()
    except Exception as e:
        print(f'[condition_number] Failed to compute condition number: {e}')
        return None


def _matrix_stats(A):
    # If the matrix is 4D (e.g. Conv2d), flatten it to 2D
    if A.ndim == 4: A = einops.rearrange(A, 'b c h w -> b (c h w)')
    # Transpose if the matrix is 'fat' (more columns than rows)
    if A.size(-2) < A.size(-1): A = A.T

    # B = (A.T @ A) - I
    B = (A.T @ A) - torch.eye(A.size(-1), device = A.device, dtype = A.dtype)

    # Frobenius norm of the deviation from the identity matrix
    # If A is perfectly orthogonal, then A @ A.T = I and ||A @ A.T - I||_F = ||B||_F 0
    lF_dev_from_eye = torch.linalg.norm(B).item()

    # Inspired by Spectral Restricted Isometry Property (SRIP) regularization:
    # Largest singular value of A^t A - I = B
    SRIP = spectral_norm(B)

    # Mutual coherence: max cosine similarity between columns (ignoring self-similarity)
    # Need to normalize for cosine sim
    A_normed = A / (A.norm(dim = 0, keepdim=True) + 1e-8)
    gram = A_normed.T @ A_normed
    gram.fill_diagonal_(0)
    mutual_coherence = gram.abs().max().item()

    return {
        'entry-mean': A.mean().item(),
        'entry-std': A.std().item(),
        'lF-norm': A.norm().item(),
        'spectral-norm': spectral_norm(A),
        'condition-number': condition_number(A),
        'lF-dev-from-eye': lF_dev_from_eye,
        'SRIP': SRIP,
        'mutual-coherence': mutual_coherence,
    }


def log_stats(model, optimizer, modules_to_log, logger, outdict):

    def RMS_to_RMS_norm_matrix(X):
        fan_out, fan_in = X.shape
        return np.sqrt(fan_in / fan_out) * spectral_norm(X)

    RMS_norm_vec = lambda x: torch.linalg.norm(x, ord = 2, dim = -1) / np.sqrt(x.shape[-1])

    for label, module in modules_to_log.items():
        if hasattr(module, 'weight'): # ignore bias

            # Stats on inputs (just to make sure they are normalized)
            outdict[f'{label}/inp-mean'].append(logger.before[(label, 'inputs')].mean().item())
            outdict[f'{label}/inp-std'].append(logger.before[(label, 'inputs')].std().item())

            # Stats on activations
            outdict[f'{label}/act-mean'].append(logger.before[(label, 'outputs')].mean().item())
            outdict[f'{label}/act-std'].append(logger.before[(label, 'outputs')].std().item())
            outdict[f'{label}/act-lF-norm'].append(logger.before[(label, 'outputs')].norm().item())
            
            # Stats on matrices (weights and grads)
            for k, v in _matrix_stats(module.weight.data).items(): outdict[f'{label}/weight-{k}'].append(v)

            if module.weight.grad is not None:
                for k, v in _matrix_stats(module.weight.grad.data).items(): outdict[f'{label}/grad-{k}'].append(v)

            # Stats on changes logged by DiffLogger
            diff_weight = logger.diff[(label, 'weights')]
            if diff_weight is not None:

                if diff_weight.ndim == 4: diff_weight = einops.rearrange(diff_weight, 'b c h w -> b (c h w)')

                w_RMS = RMS_to_RMS_norm_matrix(diff_weight).item()
                inp_RMS = RMS_norm_vec(logger.before[(label, 'inputs')])
                out_diff_RMS = RMS_norm_vec(logger.diff[(label, 'outputs')])
                outdict[f'{label}/weight-RMS'].append(w_RMS)

                if inp_RMS.shape == out_diff_RMS.shape:
                    outdict[f'{label}/RMS-mean-dist-to-bound'].append((inp_RMS * w_RMS - out_diff_RMS).mean().item())

                
            # Record stats logged by optimizer (how much applying momentum and orth change the update)
            if optimizer is None: continue
            for name, p in module.named_parameters():
                if p in optimizer.state:
                    for k, v in optimizer.state[p].items():
                        if 'stat' not in k: continue
                        k = k.replace('stat-', '')
                        outdict[f'{label}/{name}-{k}'].append(v)
