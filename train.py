# Fair comparisons of different optimizers require we tune the hyperparameters of each optimizer, 
# account

import os, copy, argparse, random
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import models, datasets, optimizers, stats

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior in CuDNN
    torch.backends.cudnn.benchmark = False     # Disables auto-tuning for max determinism

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_accuracy = float('-inf')

    def early_stop(self, validation_accuracy):
        if validation_accuracy > self.max_validation_accuracy + self.min_delta:
            self.max_validation_accuracy = validation_accuracy
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def evaluate_model(model, testloader):
    model.eval()
    device = next(model.parameters()).device
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def assign_params(model):
    # We only assign 2 or higher dim params to muon in the body of the model

    def sel_params(attr):
        out = []
        if hasattr(model, attr):
            for n, p in getattr(model, attr).named_parameters():
                if p.requires_grad:
                    out.append(p)
        return out

    muon, other = [], []
    other.extend(sel_params('head'))

    for n, p in model.body.named_parameters():
        if not p.requires_grad: continue
        if p.ndim >= 2:
            muon.append(p)

    other.extend(sel_params('tail'))
    return muon, other

def train_model(model, optims, train_loader, test_loader, epochs, modules_to_log, results,
                log_every, log_wandb = False, patience = 5, optimizer_to_log = None):

    logger = stats.DiffLogger(modules_to_log)
    print(f'Logging optimizer: {optimizer_to_log.__class__.__name__}')

    stopper = EarlyStopper(patience = patience)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_i, (inputs, labels) in enumerate(train_loader):

            inputs, labels = inputs.to(device), labels.to(device)
            inputs_copy = inputs.detach().clone()

            model.zero_grad(set_to_none=True)

            logger.clear()
            logger.set_state('before')
            logger.log_weights()

            outputs = model(inputs_copy)
            loss = criterion(outputs, labels)
            loss.backward()

            for optimizer in optims:
                optimizer.step()


            if batch_i % log_every == 0:
                with torch.no_grad():

                    logger.set_state('after')
                    logger.log_weights()
                    model(inputs_copy)  # Trigger another forward pass to logg diffs
                    stats.log_stats(model, optimizer_to_log, modules_to_log, logger, results['train_stats'])

                    
                    running_loss += loss.item()
                    results['train_stats']['main/train-loss'].append(loss.item())

                    if log_wandb:
                        wandb.log({
                            'epoch': epoch, 'batch': batch_i,
                            **{k: v[-1] for k, v in results['train_stats'].items() if 'val' not in k}
                        })

        avg_loss = running_loss / len(train_loader)
        acc = evaluate_model(model, test_loader)

        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, \tVal Accuracy={acc:.5f}, \tPatience={stopper.counter}/{patience}')
        results['train_stats']['main/val-accuracy'].append(acc)

        if log_wandb:
            wandb.log({'epoch': epoch, 'train_loss': avg_loss, 'main/val-accuracy': acc})
        
        if patience > 0 and stopper.early_stop(acc):
            print(f'Early stopping at epoch {epoch+1}')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--save', type=str, default='mnist')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--group', type=str, default='tmp')
    parser.add_argument('--log_freq', type=float, default=0.1)
    parser.add_argument('--log_optimizer_stats', action='store_true')
    parser.add_argument('--wandb', action='store_true')

    parser.add_argument('--task', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--imb_type', type=str, default=None, choices=['exp', 'step', None])
    parser.add_argument('--imb_factor', type=float, default=0.1)


    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'])
    parser.add_argument('--depth', type=int, default=2)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)

    parser.add_argument('--bn', action='store_true')
    parser.add_argument('--init_type', type=str, default='kaiming', choices=['kaiming', 'orthogonal'])
    parser.add_argument('--hidden_dim', type=int, default=128) # for MLP
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--optim', type=str, choices=['sgd', 'adam', 'muon', 'ortho'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.999])
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0)

    # Muon-specific arguments
    parser.add_argument('--muon_lr', type=float, default=0.02)
    parser.add_argument('--muon_weight_decay', type=float, default=0)
    parser.add_argument('--muon_momentum', type=float, default=0.95)
    parser.add_argument('--muon_nesterov', action='store_true')
    parser.add_argument('--ns_steps', type=int, default=5)
    
    parser.add_argument('--momentum_after_orth', action='store_true')
    parser.add_argument('--newtonschulz', action='store_true')
        
    args = parser.parse_args()
    results = defaultdict(dict)
    results['args'] = copy.deepcopy(args)

    if args.wandb:
        import wandb
        wandb.login()
        wandb.init(
            project = 'muon-project',
            group = args.group if args.group else None,
            name = args.save.split('/')[-1],
            config = results['args'],
            save_code = True
        )
    
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        # else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    set_seed(args.seed)
    print(f'device {device} with seed {args.seed}')
    
    # Load dataset
    train_ds, val_ds, input_shape, num_classes = datasets.get_datasets(args.task, imbalance_type = args.imb_type, factor = args.imb_factor)

    m_workers = min(max(torch.cuda.device_count(), 1) * 4, os.cpu_count() or 8)
    train_loader = DataLoader(train_ds, batch_size = args.batch_size, shuffle=True, num_workers=m_workers)
    val_loader   =  DataLoader(val_ds,  batch_size = args.batch_size, shuffle=False, num_workers=m_workers)
    print(f'Train / Val sizes: {len(train_ds)} / {len(val_ds)}')
    print(f'Train / Val batches: {len(train_loader)} / {len(val_loader)}')

    # Initialize model
    model = {
        'mlp': models.MLP(
            input_dim = input_shape.numel(), hidden_dim = args.hidden_dim, out_dim = num_classes,
            depth = args.depth, bn = args.bn, init_type = args.init_type,
        ).to(device),

        'cnn': models.VanillaCNN(out_dim = num_classes, init_type = args.init_type).to(device)

    }[args.model]
    print(f'Model: {model.__class__.__name__} with {args.hidden_dim} hidden dim, {args.depth} depth, and {args.init_type} init')

    # Setup optimizers
    adam_args = {'lr': args.lr, 'betas': args.betas, 'eps': args.eps, 'weight_decay': args.weight_decay}
    if args.optim in ('muon', 'ortho', 'sgd'):

        # Assign 2d and body params to orthogonalize
        orth_params, other_params = assign_params(model)
        assert orth_params, 'No orth params found in model'
        print(f'Found {len(orth_params)} orth params and {len(other_params)} other params')
        print(f'Orth param shapes: {[p.shape for p in orth_params]}')

        orth_optim_args = {
            'lr': args.muon_lr,
            'weight_decay': args.muon_weight_decay,
            'momentum': args.muon_momentum,
            'nesterov': args.muon_nesterov,
            'ns_steps': args.ns_steps,
            'log_stats': args.log_optimizer_stats,
            'newtonschulz': args.newtonschulz,
        }

        if args.optim == 'muon':
            orth_optim_args['momentum_after_orth'] = False
    
        elif args.optim == 'ortho':
            orth_optim_args['momentum_after_orth'] = True

        elif args.optim == 'sgd':
            orth_optim_args['orthogonalize'] = False

        print(f'Using optim args: {orth_optim_args}')
        opt = optimizers.MomentumOrth

        optims = [opt(orth_params, **orth_optim_args), optim.AdamW(other_params, **adam_args)]
        results['orth_group_params'] = optims[0].state_dict()['param_groups']
        results['adam_group_params'] = optims[1].state_dict()['param_groups']
    
    elif args.optim == 'adam':
        optims = [optim.AdamW(model.parameters(), **adam_args)]
        results['adam_group_params'] = optims[0].state_dict()['param_groups']
    

    # log all Linear or Conv2d layers in body
    results['train_stats'] = defaultdict(list)
    modules_to_log = {}
    for n, m in model.body.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            modules_to_log[f'body_{n}_{m.__class__.__name__}'] = m
    print(f'Logging modules: {list(modules_to_log.keys())}')

    log_every = int(args.log_freq * len(train_loader)) if args.log_freq > 0 else 1
    train_model(
        model, optims, train_loader, val_loader, args.epochs, modules_to_log,
        results, log_every = log_every, log_wandb = args.wandb, patience = args.patience,
        optimizer_to_log = optims[0] if args.log_optimizer_stats else None
    )
    
    # Save results and model
    os.makedirs(args.save, exist_ok = True)
    if args.save_model: torch.save(model.state_dict(), f'{args.save}/model.p')
    torch.save(results, f'{args.save}/results.p')
    print(f'Done! Results saved to {args.save}')