import torch
import torch.nn.functional as F
import logging
from typing import List, Tuple, Dict
import numpy as np

logger = logging.getLogger(__name__)

def _build_root_head_reference(
    server_attack_windows: List[Tuple],
    ae: torch.nn.Module,
    global_head_weights: Dict[str, torch.Tensor],
    mse_threshold: float,
    server_lr: float = 1e-3
) -> Dict[str, torch.Tensor]:
    """
    Computes the server's root reference delta for the AttackHead (Channel 2).
    
    Extracts flows from the reserved server_attack_windows (which are guaranteed
    never to have been seen by clients in train/test splits), passes them through
    the current global Autoencoder to compute reconstruction MSE and z-vectors,
    filters for high-MSE flows, and computes one gradient step on the AttackHead
    using pseudo-labels of 1.0 (since these are known attack flows).
    """
    from aura.models import AttackHead
    
    if not server_attack_windows:
        logger.warning("[ROOT] No server_attack_windows provided. Head delta will be zero.")
        return {k: torch.zeros_like(v) for k, v in global_head_weights.items()}

    # Extract all attack flows from the windows
    all_flows_list = []
    for graph, labels in server_attack_windows:
        flows = graph['edge_attr']
        attack_mask = (labels == 1)
        if attack_mask.any():
            all_flows_list.append(flows[attack_mask])
            
    if not all_flows_list:
        logger.warning("[ROOT] Server attack windows contain no attack flows.")
        return {k: torch.zeros_like(v) for k, v in global_head_weights.items()}
        
    all_flows = torch.cat(all_flows_list, dim=0)

    # Pass through AE to get z-vectors (no MSE filtering on the server)
    ae.eval()
    with torch.no_grad():
        # Process in batches to avoid OOM if there are many flows
        z_list = []
        for i in range(0, len(all_flows), 1024):
            batch = all_flows[i:i+1024]
            recon, z = ae(batch)
            z_list.append(z)
            
        all_z = torch.cat(z_list, dim=0)

    # The server knows these are attacks, so ALL of them contribute to the reference gradient
    z_for_head = all_z
    n_total = len(all_flows)
    
    print(f"[ROOT] Server Attack Flows: {n_total} total. "
          f"All {n_total} latents will contribute to Channel 2 root gradient.")

    if n_total == 0:
        logger.warning("[ROOT] No anomalous flows found in server attack windows. Returning zero delta.")
        return {k: torch.zeros_like(v) for k, v in global_head_weights.items()}

    # Initialize a fresh AttackHead with global weights
    root_head = AttackHead()
    root_head.load_state_dict(global_head_weights)
    root_head.train()
    
    head_opt = torch.optim.Adam(root_head.parameters(), lr=server_lr)
    
    # Train one full-batch step with pseudo-labels = 1.0 (malicious)
    head_opt.zero_grad()
    preds = root_head(z_for_head).squeeze()
    
    # In some edge cases z_for_head might be 1 element, causing preds to be 0-D
    if preds.dim() == 0:
        preds = preds.unsqueeze(0)
        
    pseudo_labels = torch.ones_like(preds)
    head_loss = F.binary_cross_entropy(preds, pseudo_labels)
    head_loss.backward()
    head_opt.step()
    
    root_head.eval()
    
    # Compute the mathematical delta
    root_head_delta = {
        k: root_head.state_dict()[k].clone() - global_head_weights[k]
        for k in global_head_weights
    }
    
    return root_head_delta
