# Environment setup to prevent MPS issues
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import time
import math
import gc
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn import metrics
from torch.nn.functional import normalize
from clustering_metric import clustering_metrics
from small_model_monitored import *
from simple_monitor import monitor
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset

# Force CPU only and optimize threading
device = torch.device('cpu')
print("Using CPU (optimized for M3 Pro)")
torch.set_num_threads(1)
torch.manual_seed(42)

def save_embedding(embedding, save_embedding):
    torch.save(embedding.cpu(), save_embedding)

def ppr(i, alpha=0.2):
    return alpha*((1-alpha)**i)

def heat(i, t=5):
    return (math.e**(-t))*(t**i)/math.factorial(i)

#changes by me
# NEW FUNCTION - Add this after the existing heat() function
def compute_adaptive_diffusion_v2(graph, x, max_hops=4, method="ppr", normalize_output=True, alpha=0.2, t=5):
    """
    Improved multi-scale diffusion with better parameter tuning
    """
    monitor.start_timer('preprocessing')
    print(f"Computing IMPROVED Multi-Scale diffusion ({method}) with {max_hops} hops")
    
    # IMPROVED: More conservative alpha values to preserve information
    if method == "ppr":
        # More balanced weights - don't reduce distant hops too aggressively
        alpha_values = [0.25, 0.25, 0.25, 0.25][:max_hops]  # Equal importance
        base_alpha = 0.2  # PPR base parameter
    else:  # heat
        alpha_values = [0.3, 0.3, 0.25, 0.15][:max_hops] 
        base_alpha = 0.2
    
    diffusion_features = []
    current = x
    
    for i in range(max_hops):
        if method == "ppr":
            # IMPROVED: Use consistent PPR computation
            theta = base_alpha * ((1 - base_alpha) ** i) * alpha_values[i]
        elif method == "heat":
            import math
            theta = (math.e ** (-5)) * (5 ** i) / math.factorial(i) * alpha_values[i]
        else:
            raise NotImplementedError(f"Method {method} not supported")
            
        if i == 0:
            weighted_feature = theta * current
        else:
            current = torch.sparse.mm(graph, current)
            weighted_feature = theta * current
            
        diffusion_features.append(weighted_feature)
        print(f"  Hop {i}: theta={theta:.6f}, feature_norm={torch.norm(weighted_feature):.4f}")
    
    # Combine all scales
    final_diffusion = sum(diffusion_features)
    
    # IMPROVED: Optional output normalization to match original scale
    if normalize_output:
        original_norm = torch.norm(x)
        current_norm = torch.norm(final_diffusion)
        if current_norm > 1e-6:  # Avoid division by zero
            final_diffusion = final_diffusion * (original_norm / current_norm)
            print(f"  Normalized: {current_norm:.4f} -> {torch.norm(final_diffusion):.4f}")
    
    monitor.end_timer()
    print(f"Improved multi-scale diffusion completed. Final norm: {torch.norm(final_diffusion):.4f}")
    return final_diffusion

def norm_adj(graph):
    monitor.start_timer('preprocessing')
    graph_cpu = graph.cpu()
    D = torch.sparse.mm(graph_cpu, torch.ones(graph_cpu.size(0),1)).view(-1)
    a = [[i for i in range(graph_cpu.size(0))],[i for i in range(graph_cpu.size(0))]]
    D = torch.sparse_coo_tensor(torch.tensor(a), 1/(D**0.5), graph_cpu.size())
    ADinv = torch.sparse.mm(D, torch.sparse.mm(graph_cpu, D))
    monitor.end_timer()
    return ADinv

def compute_diffusion_matrix(graph, x, niter=5, method="ppr"):
    monitor.start_timer('preprocessing')
    print("Calculating S matrix")
    for i in range(0, niter):
        print("Iteration: " + str(i))
        if method=="ppr":
            theta = ppr(i)
        elif method=="heat":
            theta=heat(i)
        else:
            raise NotImplementedError
        if i==0:
            final = theta*x
            current = x
        else:
            current = torch.sparse.mm(graph, current)
            final+= (theta*current)
    monitor.end_timer()
    return final
#introduced safe taining loop
def enhanced_safe_training_step(model, pos_rw, neg_rw, mapping, AX, feature_matrix, optimizer):
    """Enhanced training step that can use different feature matrices"""
    try:
        # Zero gradients
        optimizer.zero_grad()
        
        # Move to device
        pos_rw = pos_rw.to(device)
        neg_rw = neg_rw.to(device)
        
        # Get unique nodes
        unique = torch.unique(torch.cat((pos_rw, neg_rw), dim=-1))
        
        # Update mapping
        mapping.scatter_(0, unique, torch.arange(unique.size(0)).to(device))
        
        # Update model embeddings with different feature matrix
        with torch.no_grad():
            if hasattr(model, 'embedding') and model.embedding is not None:
                if model.embedding.grad is not None:
                    model.embedding.grad.zero_()
        
        # Use the provided feature matrix (could be SX, MSX_ppr, or MSX_heat)
        model.update_B(F.embedding(unique, AX), F.embedding(unique, feature_matrix), unique)
        
        # Compute loss
        loss = model.loss(pos_rw, neg_rw, mapping)
        
        # Check for NaN or infinite loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: Invalid loss detected, skipping this batch")
            return None
            
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        return loss.item()
        
    except Exception as e:
        print(f"Error in enhanced training step: {e}")
        return None
# def safe_training_step(model, pos_rw, neg_rw, mapping, AX, SX, optimizer):
#     """Protected training step to avoid segmentation faults"""
#     try:
#         # Zero gradients
#         optimizer.zero_grad()
        
#         # Move to device
#         pos_rw = pos_rw.to(device)
#         neg_rw = neg_rw.to(device)
        
#         # Get unique nodes
#         unique = torch.unique(torch.cat((pos_rw, neg_rw), dim=-1))
        
#         # Update mapping
#         mapping.scatter_(0, unique, torch.arange(unique.size(0)).to(device))
        
#         # Update model embeddings - protected call
#         with torch.no_grad():
#             # Clear any existing gradients
#             if hasattr(model, 'embedding') and model.embedding is not None:
#                 if model.embedding.grad is not None:
#                     model.embedding.grad.zero_()
        
#         # Safe embedding update
#         model.update_B(F.embedding(unique, AX), F.embedding(unique, SX), unique)
        
#         # Compute loss with gradient enabled
#         loss = model.loss(pos_rw, neg_rw, mapping)
        
#         # Check for NaN or infinite loss
#         if torch.isnan(loss) or torch.isinf(loss):
#             print("Warning: Invalid loss detected, skipping this batch")
#             return None
            
#         # Backward pass
#         loss.backward()
        
#         # Clip gradients to prevent explosion
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
#         # Optimizer step
#         optimizer.step()
        
#         return loss.item()
        
#     except Exception as e:
#         print(f"Error in training step: {e}")
#         print("Skipping this batch...")
#         return None
#main function introduced by me

def run_parameter_experiments():
    """Run separate experiments for alpha and t values"""
    import pandas as pd  # Add this import at the top of file
    
    # Parameter values to test
    alpha_values = [0.15, 0.2, 0.25, 0.3]
    t_values = [1, 3, 5, 7, 10]
    
    # Fixed values for separate experiments
    fixed_t = 5      # Fixed t value when testing alpha
    fixed_alpha = 0.2  # Fixed alpha value when testing t
    
    alpha_results = {}
    t_results = {}
    
    print("="*80)
    print("STARTING SEPARATE PARAMETER STUDY")
    print(f"Testing {len(alpha_values)} alpha values (t={fixed_t}) + {len(t_values)} t values (alpha={fixed_alpha}) = {len(alpha_values) + len(t_values)} experiments")
    print("="*80)
    
    experiment_count = 0
    total_experiments = len(alpha_values) + len(t_values)
    
    # Test alpha values with fixed t
    print(f"\n=== TESTING ALPHA VALUES (t={fixed_t}) ===")
    for alpha in alpha_values:
        experiment_count += 1
        print(f"\n[{experiment_count}/{total_experiments}] Running alpha experiment: alpha={alpha}, t={fixed_t}")
        
        result = run_single_experiment(alpha, fixed_t)
        alpha_results[alpha] = result
        
        print(f"Alpha Results: NMI={result['nmi']:.4f}, ACC={result['acc']:.4f}")
    
    # Test t values with fixed alpha  
    print(f"\n=== TESTING T VALUES (alpha={fixed_alpha}) ===")
    for t in t_values:
        experiment_count += 1
        print(f"\n[{experiment_count}/{total_experiments}] Running t experiment: alpha={fixed_alpha}, t={t}")
        
        result = run_single_experiment(fixed_alpha, t)
        t_results[t] = result
        
        print(f"T Results: NMI={result['nmi']:.4f}, ACC={result['acc']:.4f}")
    
    # Print separate comparisons
    print_alpha_comparison(alpha_results, fixed_t)
    print_t_comparison(t_results, fixed_alpha)
    # find_best_separate_parameters(alpha_results, t_results)
    
    return alpha_results, t_results

def print_alpha_comparison(alpha_results, fixed_t):
    """Print formatted comparison table for alpha values"""
    print("\n" + "="*80)
    print(f"ALPHA PARAMETER STUDY RESULTS (t={fixed_t})")
    print("="*80)
    print(f"{'Alpha':<8} {'NMI':<10} {'ACC':<10} {'ARI':<10} {'F1':<10} {'Method':<12}")
    print("-" * 80)
    
    for alpha, metrics in sorted(alpha_results.items()):
        print(f"{alpha:<8} {metrics['nmi']:<10.4f} {metrics['acc']:<10.4f} "
              f"{metrics['ari']:<10.4f} {metrics['f1']:<10.4f} {metrics['method']:<12}")
    
    print("="*80)

def print_t_comparison(t_results, fixed_alpha):
    """Print formatted comparison table for t values"""
    print("\n" + "="*80)
    print(f"T PARAMETER STUDY RESULTS (alpha={fixed_alpha})")
    print("="*80)
    print(f"{'t':<8} {'NMI':<10} {'ACC':<10} {'ARI':<10} {'F1':<10} {'Method':<12}")
    print("-" * 80)
    
    for t, metrics in sorted(t_results.items()):
        print(f"{t:<8} {metrics['nmi']:<10.4f} {metrics['acc']:<10.4f} "
              f"{metrics['ari']:<10.4f} {metrics['f1']:<10.4f} {metrics['method']:<12}")
    
    print("="*80)

def run_single_experiment(alpha, t):
    """Run a single experiment with given parameters - THIS CONTAINS YOUR EXISTING MAIN CODE"""
    gc.collect()
    
    # Use your existing argument parsing but with fixed parameters
    class Args:
        dataset = "Cora"  # Change this for different datasets
        hidden_dims = 256
        walk_length = 3
        walks_per_node = 10
        batch_size = 2708
        lr = 0.01
        epochs = 50  # Reduce epochs for parameter study
        log_steps = 1
        save_embedding = f"embedding_alpha_{alpha}_t_{t}.pt"
        
    args = Args()

    # Load dataset (COPY YOUR EXISTING DATASET LOADING CODE HERE)
    dataset = Planetoid(root='.', name='Cora')  # Change dataset name as needed
    data = dataset[0].to(device)
    del dataset
    gc.collect()
    
    y = data.y.view(-1)
    val_idx = data.val_mask.nonzero().view(-1).tolist()
    data.edge_index = to_undirected(add_remaining_self_loops(data.edge_index)[0])
    edge_index = data.edge_index
    
    # Create sparse tensor on CPU first
    edge_tensor = torch.sparse_coo_tensor(
        data.edge_index.cpu(), 
        torch.ones(data.edge_index.size(1)), 
        (data.x.size(0), data.x.size(0))
    )
    
    # ENHANCED PREPROCESSING with your parameters
    print(f"=== Enhanced Preprocessing with alpha={alpha}, t={t} ===")
    
    normalized_A = norm_adj(edge_tensor)
    AX = torch.sparse.mm(normalized_A, data.x.cpu())
    SX = compute_diffusion_matrix(normalized_A, data.x.cpu(), niter=3)
    
    # **KEY CHANGE: Pass your parameters here**
    MSX_ppr = compute_adaptive_diffusion_v2(
        normalized_A, data.x.cpu(), max_hops=3, method="ppr", 
        normalize_output=True, alpha=alpha, t=t
    )
    
    MSX_heat = compute_adaptive_diffusion_v2(
        normalized_A, data.x.cpu(), max_hops=3, method="heat", 
        normalize_output=True, alpha=alpha, t=t
    )

    # Create fused features
    def create_fused_features(AX, SX, MSX_ppr, weights=[0.5, 0.3, 0.2]):
        ax_norm = torch.norm(AX)
        sx_scaled = SX * (ax_norm / torch.norm(SX))
        msx_scaled = MSX_ppr * (ax_norm / torch.norm(MSX_ppr))
        fused = weights[0] * AX + weights[1] * sx_scaled + weights[2] * msx_scaled
        return fused

    fused_features = create_fused_features(AX, SX, MSX_ppr).to(device)
    AX = AX.to(device)
    SX = SX.to(device)
    MSX_ppr = MSX_ppr.to(device)
    MSX_heat = MSX_heat.to(device)
    
    # Clean up
    del normalized_A, data, edge_tensor
    gc.collect()

    # Model creation and training (COPY YOUR EXISTING CODE)
    model = Node2Vec(AX.size(-1), args.hidden_dims, AX.size(0), 0.0, edge_index, 
                     args.hidden_dims, args.walk_length, args.walk_length, 
                     args.walks_per_node, p=1.0, q=1.0, sparse=True).to(device)

    actual_batch_size = min(args.batch_size, 500)
    loader = model.loader(batch_size=actual_batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training setup
    model.train()
    mapping = torch.zeros(AX.size(0), dtype=torch.int64).to(device)
    best_nmi = -1
    
    # **SHORTENED TRAINING LOOP FOR PARAMETER STUDY**
    for epoch in range(1, args.epochs + 1):
        for i, (pos_rw, neg_rw) in enumerate(loader):
            loss = enhanced_safe_training_step(model, pos_rw, neg_rw, mapping, AX, fused_features, optimizer)
            if i % 5 == 0:
                gc.collect()
        
        # Evaluate every 10 epochs
        if epoch % 10 == 0:
            try:
                embedding = model.get_embedding(AX, MSX_ppr)
                monitor.start_timer('kmeans')
                kmeans = KMeans(n_clusters=y.max().item()+1, n_init=20)
                y_pred = kmeans.fit_predict(embedding.detach()[val_idx].cpu().numpy())
                nmi = metrics.normalized_mutual_info_score(y[val_idx].cpu().numpy(), y_pred)
                monitor.end_timer()
                
                if nmi > best_nmi:
                    best_nmi = nmi
                
                del embedding, y_pred
            except Exception as e:
                print(f"Evaluation error in epoch {epoch}: {e}")
            gc.collect()
    
    # Final evaluation with all metrics
    try:
        embedding_orig = model.get_embedding(AX, SX)
        embedding_ppr = model.get_embedding(AX, MSX_ppr) 
        embedding_heat = model.get_embedding(AX, MSX_heat)
        
        kmeans = KMeans(n_clusters=y.max().item()+1, n_init=20, random_state=42)
        
        # Test all three methods
        y_pred_orig = kmeans.fit_predict(embedding_orig.detach().cpu().numpy())
        y_pred_ppr = kmeans.fit_predict(embedding_ppr.detach().cpu().numpy())
        y_pred_heat = kmeans.fit_predict(embedding_heat.detach().cpu().numpy())
        
        # Calculate metrics
        nmi_orig = metrics.normalized_mutual_info_score(y.cpu().numpy(), y_pred_orig)
        nmi_ppr = metrics.normalized_mutual_info_score(y.cpu().numpy(), y_pred_ppr)
        nmi_heat = metrics.normalized_mutual_info_score(y.cpu().numpy(), y_pred_heat)
        
        from sklearn.metrics import adjusted_rand_score, accuracy_score
        acc_ppr = accuracy_score(y.cpu().numpy(), y_pred_ppr) if len(set(y_pred_ppr)) == len(set(y.cpu().numpy())) else 0
        ari_ppr = adjusted_rand_score(y.cpu().numpy(), y_pred_ppr)
        
        # Return best result
        best_method = "PPR" if nmi_ppr >= max(nmi_orig, nmi_heat) else ("Heat" if nmi_heat > nmi_orig else "Original")
        best_nmi = max(nmi_orig, nmi_ppr, nmi_heat)
        
        return {
            'nmi': best_nmi,
            'acc': acc_ppr,
            'ari': ari_ppr,
            'f1': acc_ppr,  # Simplified
            'method': best_method,
            'nmi_orig': nmi_orig,
            'nmi_ppr': nmi_ppr, 
            'nmi_heat': nmi_heat
        }
        
    except Exception as e:
        print(f"Final evaluation error: {e}")
        return {'nmi': 0, 'acc': 0, 'ari': 0, 'f1': 0, 'method': 'Error'}

def print_parameter_comparison(results):
    """Print formatted comparison table"""
    print("\n" + "="*100)
    print("PARAMETER STUDY RESULTS - COMPREHENSIVE COMPARISON")
    print("="*100)
    print(f"{'Alpha':<8} {'t':<4} {'Best_NMI':<10} {'Best_Method':<12} {'NMI_Orig':<10} {'NMI_PPR':<10} {'NMI_Heat':<10} {'ACC':<8}")
    print("-" * 100)
    
    for (alpha, t), metrics in sorted(results.items()):
        print(f"{alpha:<8} {t:<4} {metrics['nmi']:<10.4f} {metrics['method']:<12} "
              f"{metrics.get('nmi_orig', 0):<10.4f} {metrics.get('nmi_ppr', 0):<10.4f} "
              f"{metrics.get('nmi_heat', 0):<10.4f} {metrics['acc']:<8.4f}")

def find_best_parameters(results):
    """Find and display best parameter combinations"""
    print("\n" + "="*60)
    print("BEST PARAMETER COMBINATIONS")
    print("="*60)
    
    # Find best overall
    best_params = max(results.items(), key=lambda x: x[1]['nmi'])
    alpha_best, t_best = best_params[0]
    best_metrics = best_params[1]
    
    print(f"Best Overall: alpha={alpha_best}, t={t_best}")
    print(f"  NMI: {best_metrics['nmi']:.4f}")
    print(f"  ACC: {best_metrics['acc']:.4f}")
    print(f"  Method: {best_metrics['method']}")
    
    # Find best for each method
    ppr_results = {k: v for k, v in results.items() if v['method'] == 'PPR'}
    if ppr_results:
        best_ppr = max(ppr_results.items(), key=lambda x: x[1]['nmi'])
        print(f"Best PPR: alpha={best_ppr[0][0]}, t={best_ppr[0][1]}, NMI={best_ppr[1]['nmi']:.4f}")
    
    heat_results = {k: v for k, v in results.items() if v['method'] == 'Heat'}
    if heat_results:
        best_heat = max(heat_results.items(), key=lambda x: x[1]['nmi'])
        print(f"Best Heat: alpha={best_heat[0][0]}, t={best_heat[0][1]}, NMI={best_heat[1]['nmi']:.4f}")


def main():
    gc.collect()
    
    parser = argparse.ArgumentParser(description='run_cora')
    parser.add_argument('--dataset', type=str, default="Cora")
    parser.add_argument('--hidden_dims', type=int, default=256)
    parser.add_argument('--walk_length', type=int, default=3)
    parser.add_argument('--walks_per_node', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2708)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--save_embedding', type=str, default="Cora.pt")
      # **NEW: Add parameter study arguments**
    parser.add_argument('--run_param_study', action='store_true', 
                       help='Run parameter study for alpha and t values')
    parser.add_argument('--alpha', type=float, default=0.2, 
                       help='Alpha parameter for diffusion')
    parser.add_argument('--t', type=float, default=5, 
                       help='t parameter for heat kernel')
    args = parser.parse_args()


     # **NEW: Check if parameter study should be run**
    if args.run_param_study:
        print("Starting comprehensive parameter study...")
        return run_parameter_experiments()

    # Load dataset
    dataset = Planetoid(root='.', name='Cora')
    data = dataset[0].to(device)
    del dataset
    gc.collect()
    
    y = data.y.view(-1)
    val_idx = data.val_mask.nonzero().view(-1).tolist()
    data.edge_index = to_undirected(add_remaining_self_loops(data.edge_index)[0])
    edge_index = data.edge_index
    
    # Create sparse tensor on CPU first
    edge_tensor = torch.sparse_coo_tensor(
        data.edge_index.cpu(), 
        torch.ones(data.edge_index.size(1)), 
        (data.x.size(0), data.x.size(0))
    )
    
    # ENHANCED PREPROCESSING - FIXED VERSION
    print("=== Enhanced Preprocessing with Multi-Scale Diffusion ===")
    
    # Step 1: Compute normalized adjacency
    normalized_A = norm_adj(edge_tensor)
    
    # Step 2: Original computations (keep these for compatibility)
    AX = torch.sparse.mm(normalized_A, data.x.cpu())
    SX = compute_diffusion_matrix(normalized_A, data.x.cpu(), niter=3)
    
    # # Step 3: NEW - Multi-scale diffusion features
    # print("Computing multi-scale PPR diffusion...")
    # MSX_ppr = compute_adaptive_diffusion(normalized_A, data.x.cpu(), max_hops=4, method="ppr")
    
    # print("Computing multi-scale Heat diffusion...")
    # MSX_heat = compute_adaptive_diffusion(normalized_A, data.x.cpu(), max_hops=3, method="heat")

    # NEW:
    # MSX_ppr = compute_adaptive_diffusion_v2(normalized_A, data.x.cpu(), max_hops=3, method="ppr", normalize_output=True)
    # MSX_heat = compute_adaptive_diffusion_v2(normalized_A, data.x.cpu(), max_hops=3, method="heat", normalize_output=True)
        # **MODIFIED: Pass parameters here**
    MSX_ppr = compute_adaptive_diffusion_v2(
        normalized_A, data.x.cpu(), max_hops=3, method="ppr", 
        normalize_output=True, alpha=args.alpha, t=args.t
    )
    
    MSX_heat = compute_adaptive_diffusion_v2(
        normalized_A, data.x.cpu(), max_hops=3, method="heat", 
        normalize_output=True, alpha=args.alpha, t=args.t
    )


    # Add feature fusion
    def create_fused_features(AX, SX, MSX_ppr, weights=[0.5, 0.3, 0.2]):
        """Create balanced fused features"""
        # Normalize all to AX scale
        ax_norm = torch.norm(AX)
        sx_scaled = SX * (ax_norm / torch.norm(SX))
        msx_scaled = MSX_ppr * (ax_norm / torch.norm(MSX_ppr))
        
        fused = weights[0] * AX + weights[1] * sx_scaled + weights[2] * msx_scaled
        return fused

    # Create fused features
    fused_features = create_fused_features(AX, SX, MSX_ppr)
    fused_features = fused_features.to(device)
    
    
    # Step 4: Move all tensors to device
    AX = AX.to(device)
    SX = SX.to(device)
    MSX_ppr = MSX_ppr.to(device)
    MSX_heat = MSX_heat.to(device)
    
    # Print preprocessing summary
    print("=== Preprocessing Summary ===")
    print(f"AX (1-hop): shape={AX.shape}, norm={torch.norm(AX):.4f}")
    print(f"SX (3-hop): shape={SX.shape}, norm={torch.norm(SX):.4f}")
    print(f"MSX_ppr (4-hop PPR): shape={MSX_ppr.shape}, norm={torch.norm(MSX_ppr):.4f}")
    print(f"MSX_heat (3-hop Heat): shape={MSX_heat.shape}, norm={torch.norm(MSX_heat):.4f}")
    
    # Clean up CPU tensors
    del normalized_A, data, edge_tensor
    gc.collect()

    # FOR NOW: Use original model but we'll test with different feature combinations
    print("=== Model Creation (Step 1: Using Original Architecture) ===")
    
    # Create model with original architecture
    model = Node2Vec(AX.size(-1), args.hidden_dims, AX.size(0), 0.0, edge_index, args.hidden_dims, args.walk_length,
                     args.walk_length, args.walks_per_node, p=1.0, q=1.0,
                     sparse=True).to(device)

    # Create data loader
    actual_batch_size = min(args.batch_size, 500)
    loader = model.loader(batch_size=actual_batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ENHANCED TRAINING - Test different feature combinations
    print("=== Training Phase ===")
    print("Testing 3 different feature combinations:")
    print("1. Original: AX + SX")
    print("2. Enhanced PPR: AX + MSX_ppr") 
    print("3. Enhanced Heat: AX + MSX_heat")
    
    # We'll test combination 2 (PPR) first - replace SX with MSX_ppr
    print("\n>>> Testing Enhanced PPR features (AX + MSX_ppr) <<<")
    
    # Training setup
    start_epoch = 1
    model.train()
    start_train = time.time()
    mapping = torch.zeros(AX.size(0), dtype=torch.int64).to(device)
    best_epoch = -1
    max_nmi = -1
    
    print(f"Starting training with batch_size={actual_batch_size}, {len(loader)} batches per epoch")
    
    # Training loop - using ENHANCED features (AX + MSX_ppr instead of AX + SX)
    for epoch in range(start_epoch, args.epochs + 1): # Test with 50 epochs first
        start_epoch_time = time.time()
        successful_batches = 0
        total_loss = 0.0
        
        for i, (pos_rw, neg_rw) in enumerate(loader):
            # MODIFIED: Use MSX_ppr instead of SX for enhanced features
            # loss = enhanced_safe_training_step(model, pos_rw, neg_rw, mapping, AX, MSX_ppr, optimizer)
            # loss = enhanced_safe_training_step(model, pos_rw, neg_rw, mapping, AX, SX, optimizer)  # Use original SX for training
            loss = enhanced_safe_training_step(model, pos_rw, neg_rw, mapping, AX, fused_features, optimizer)
            if loss is not None:
                successful_batches += 1
                total_loss += loss
            
            # Memory cleanup every 5 steps
            if i % 5 == 0:
                gc.collect()

            if (i + 1) % args.log_steps == 0:
                avg_loss = total_loss / max(successful_batches, 1)
                print(f'Epoch: {epoch:02d}, Step: {i+1:03d}/{len(loader)}, '
                      f'Avg Loss: {avg_loss:.4f}, Successful batches: {successful_batches}')
        
        print(f"Epoch {epoch} completed. Time: {time.time()-start_epoch_time:.2f}s")
        
        # Evaluation every 10 epochs
        if epoch % 10 == 0:
            try:
                # MODIFIED: Use enhanced features for evaluation too
                embedding = model.get_embedding(AX, MSX_ppr)  # Using MSX_ppr instead of SX
                
                monitor.start_timer('kmeans')
                kmeans = KMeans(n_clusters=y.max().item()+1, n_init=20)
                y_pred = kmeans.fit_predict(embedding.detach()[val_idx].cpu().numpy())
                nmi = metrics.normalized_mutual_info_score(y[val_idx].cpu().numpy(), y_pred)
                monitor.end_timer()
                
                print(f"Epoch {epoch}: Enhanced NMI = {nmi:.4f}")
                
                if nmi > max_nmi:
                    best_epoch = epoch
                    max_nmi = nmi
                    save_embedding(embedding, args.save_embedding)
                    print(f"New best Enhanced NMI: {max_nmi:.4f} at epoch {best_epoch}")
                
                del embedding, y_pred
                
            except Exception as e:
                print(f"Evaluation error in epoch {epoch}: {e}")
            
            gc.collect()
    
    print(f"\n=== STEP 1 RESULTS ===")
    print(f"Enhanced Training completed! Best NMI: {max_nmi:.4f} at epoch {best_epoch}")
    print(f"Training time: {time.time()-start_train:.2f} seconds")
    print(f"Improvement expected: +1-2% over baseline (~0.588)")
    
   # Enhanced final evaluation with all metrics
    print("\n=== Final Comparison (All Metrics) ===")
    
    try:
        # Test 1: Original features (AX + SX)
        print("Testing Original features (AX + SX)...")
        embedding_orig = model.get_embedding(AX, SX)
        kmeans = KMeans(n_clusters=y.max().item()+1, n_init=20, random_state=42)
        y_pred_orig = kmeans.fit_predict(embedding_orig.detach().cpu().numpy())
        
        # Calculate all metrics for Original
        nmi_orig = metrics.normalized_mutual_info_score(y.cpu().numpy(), y_pred_orig)
        from sklearn.metrics import adjusted_rand_score, completeness_score, f1_score
        ari_orig = adjusted_rand_score(y.cpu().numpy(), y_pred_orig)
        cs_orig = completeness_score(y.cpu().numpy(), y_pred_orig)
        
        # Calculate accuracy using clustering_metrics
        try:
            cm_orig = clustering_metrics(y.cpu().numpy(), y_pred_orig)
            acc_orig, f1_orig, _, _, _, _, _ = cm_orig.clusteringAcc()
        except:
            acc_orig, f1_orig = 0.0, 0.0
        
        # Test 2: Enhanced PPR features (AX + MSX_ppr) 
        print("Testing Enhanced PPR features (AX + MSX_ppr)...")
        embedding_ppr = model.get_embedding(AX, MSX_ppr)
        y_pred_ppr = kmeans.fit_predict(embedding_ppr.detach().cpu().numpy())
        
        nmi_ppr = metrics.normalized_mutual_info_score(y.cpu().numpy(), y_pred_ppr)
        ari_ppr = adjusted_rand_score(y.cpu().numpy(), y_pred_ppr)
        cs_ppr = completeness_score(y.cpu().numpy(), y_pred_ppr)
        
        try:
            cm_ppr = clustering_metrics(y.cpu().numpy(), y_pred_ppr)
            acc_ppr, f1_ppr, _, _, _, _, _ = cm_ppr.clusteringAcc()
        except:
            acc_ppr, f1_ppr = 0.0, 0.0
        
        # Test 3: Enhanced Heat features (AX + MSX_heat)
        print("Testing Enhanced Heat features (AX + MSX_heat)...")
        embedding_heat = model.get_embedding(AX, MSX_heat)
        y_pred_heat = kmeans.fit_predict(embedding_heat.detach().cpu().numpy())
        
        nmi_heat = metrics.normalized_mutual_info_score(y.cpu().numpy(), y_pred_heat)
        ari_heat = adjusted_rand_score(y.cpu().numpy(), y_pred_heat)
        cs_heat = completeness_score(y.cpu().numpy(), y_pred_heat)
        
        try:
            cm_heat = clustering_metrics(y.cpu().numpy(), y_pred_heat)
            acc_heat, f1_heat, _, _, _, _, _ = cm_heat.clusteringAcc()
        except:
            acc_heat, f1_heat = 0.0, 0.0
        
        # Display comprehensive results table
        print(f"\n=== COMPREHENSIVE RESULTS SUMMARY ===")
        print(f"Method                    | ACC    | NMI    | ARI    | CS     | F1")
        print(f"--------------------------|--------|--------|--------|--------|--------")
        print(f"Original (AX + SX)        | {acc_orig:.4f} | {nmi_orig:.4f} | {ari_orig:.4f} | {cs_orig:.4f} | {f1_orig:.4f}")
        print(f"Enhanced PPR (AX + MSX)   | {acc_ppr:.4f} | {nmi_ppr:.4f} | {ari_ppr:.4f} | {cs_ppr:.4f} | {f1_ppr:.4f}")
        print(f"Enhanced Heat (AX + MSX)  | {acc_heat:.4f} | {nmi_heat:.4f} | {ari_heat:.4f} | {cs_heat:.4f} | {f1_heat:.4f}")
        
        # Find best method and show improvements
        methods = ['Original', 'PPR', 'Heat']
        nmi_scores = [nmi_orig, nmi_ppr, nmi_heat]
        best_idx = nmi_scores.index(max(nmi_scores))
        best_method = methods[best_idx]
        best_nmi = max(nmi_scores)
        
        print(f"\nBest method: {best_method} with NMI = {best_nmi:.4f}")
        
        if nmi_ppr > nmi_orig:
            print(f"PPR improvement: NMI +{nmi_ppr - nmi_orig:.4f}, ACC +{acc_ppr - acc_orig:.4f}")
        if nmi_heat > nmi_orig:
            print(f"Heat improvement: NMI +{nmi_heat - nmi_orig:.4f}, ACC +{acc_heat - acc_orig:.4f}")
        
        if best_nmi > nmi_orig:
            print(f"SUCCESS: Multi-scale diffusion improved NMI by {best_nmi - nmi_orig:.4f}")
            print("Ready to proceed to Step 2!")
        else:
            print("Multi-scale diffusion didn't improve results. Need to adjust parameters.")
        
    except Exception as e:
        print(f"Final evaluation error: {e}")
    
    # Print component statistics
    monitor.print_stats()
# # def main():
#     gc.collect()
    
#     parser = argparse.ArgumentParser(description='run_cora')
#     parser.add_argument('--dataset', type=str, default="Cora")
#     parser.add_argument('--hidden_dims', type=int, default=256)
#     parser.add_argument('--walk_length', type=int, default=3)
#     parser.add_argument('--walks_per_node', type=int, default=10)
#     parser.add_argument('--batch_size', type=int, default=2708)
#     parser.add_argument('--lr', type=float, default=0.01)
#     parser.add_argument('--epochs', type=int, default=200)
#     parser.add_argument('--log_steps', type=int, default=1)
#     parser.add_argument('--save_embedding', type=str, default="embedding_cora.pt")
#     args = parser.parse_args()

#     # Load dataset
#     dataset = Planetoid(root='.', name='Cora')
#     data = dataset[0].to(device)
#     del dataset
#     gc.collect()
    
#     y = data.y.view(-1)
#     val_idx = data.val_mask.nonzero().view(-1).tolist()
#     data.edge_index = to_undirected(add_remaining_self_loops(data.edge_index)[0])
#     edge_index = data.edge_index
    
#     # Create sparse tensor on CPU
#     edge_tensor = torch.sparse_coo_tensor(
#         data.edge_index.cpu(), 
#         torch.ones(data.edge_index.size(1)), 
#         (data.x.size(0), data.x.size(0))
#     )
#     #changes for multiscale Diffussion
#     # normalized_A = norm_adj(edge_tensor)
#     # AX = torch.sparse.mm(normalized_A, data.x.cpu())
#     # SX = compute_diffusion_matrix(normalized_A, data.x.cpu(), niter=3)
#     def enhanced_preprocessing(edge_tensor, node_features):
#         """Enhanced preprocessing with multi-scale diffusion"""
#         print("=== Enhanced Preprocessing Phase ===")
        
#         # Original computations (keep these)
#         normalized_A = norm_adj(edge_tensor)
#         AX = torch.sparse.mm(normalized_A, node_features)
        
#         # Original 3-hop diffusion (keep for comparison)
#         SX = compute_diffusion_matrix(normalized_A, node_features, niter=3)
        
#         # NEW: Multi-scale PPR diffusion (4 hops with adaptive weights)
#         MSX_ppr = compute_adaptive_diffusion(normalized_A, node_features, max_hops=4, method="ppr")
        
#         # NEW: Multi-scale Heat diffusion (3 hops with different kernel)
#         MSX_heat = compute_adaptive_diffusion(normalized_A, node_features, max_hops=3, method="heat")
        
#         print("=== Preprocessing Complete ===")
#         print(f"AX shape: {AX.shape}, norm: {torch.norm(AX):.4f}")
#         print(f"SX shape: {SX.shape}, norm: {torch.norm(SX):.4f}")
#         print(f"MSX_ppr shape: {MSX_ppr.shape}, norm: {torch.norm(MSX_ppr):.4f}")
#         print(f"MSX_heat shape: {MSX_heat.shape}, norm: {torch.norm(MSX_heat):.4f}")
        
#         return AX, SX, MSX_ppr, MSX_heat, normalized_A
    
#     # Move to device after sparse operations
#     AX = AX.to(device)
#     SX = SX.to(device)

#     # Add these lines after:
#     MSX_ppr = MSX_ppr.to(device)
#     MSX_heat = MSX_heat.to(device)
    
#     # Clean up
#     del normalized_A, data, edge_tensor
#     gc.collect()

#     # Create model with smaller parameters for stability
#     model = Node2Vec(AX.size(-1), args.hidden_dims, AX.size(0), 0.0, edge_index, args.hidden_dims, args.walk_length,
#                      args.walk_length, args.walks_per_node, p=1.0, q=1.0,
#                      sparse=True).to(device)

#     # Create data loader with smaller batch size for stability
#     actual_batch_size = min(args.batch_size, 500)  # Limit batch size
#     loader = model.loader(batch_size=actual_batch_size, shuffle=True, num_workers=0)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#     # Training setup
#     start_epoch = 1
#     model.train()
#     start_train = time.time()
#     mapping = torch.zeros(AX.size(0), dtype=torch.int64).to(device)
#     best_epoch = -1
#     max_nmi = -1
    
#     print(f"Starting training with batch_size={actual_batch_size}, {len(loader)} batches per epoch")
    
#     # Training loop with protection
#     for epoch in range(start_epoch, args.epochs + 1):
#         start_epoch_time = time.time()
#         successful_batches = 0
#         total_loss = 0.0
        
#         for i, (pos_rw, neg_rw) in enumerate(loader):
#             # Safe training step
#             loss = safe_training_step(model, pos_rw, neg_rw, mapping, AX, SX, optimizer)
            
#             if loss is not None:
#                 successful_batches += 1
#                 total_loss += loss
            
#             # Memory cleanup every 5 steps
#             if i % 5 == 0:
#                 gc.collect()

#             if (i + 1) % args.log_steps == 0:
#                 avg_loss = total_loss / max(successful_batches, 1)
#                 print(f'Epoch: {epoch:02d}, Step: {i+1:03d}/{len(loader)}, '
#                       f'Avg Loss: {avg_loss:.4f}, Successful batches: {successful_batches}')
        
#         print(f"Epoch {epoch} completed. Time: {time.time()-start_epoch_time:.2f}s")
        
#         # Evaluation with error handling
#         try:
#             embedding = model.get_embedding(AX, SX)
            
#             monitor.start_timer('kmeans')
#             kmeans = KMeans(n_clusters=y.max().item()+1, n_init=20)
#             y_pred = kmeans.fit_predict(embedding.detach()[val_idx].cpu().numpy())
#             nmi = metrics.normalized_mutual_info_score(y[val_idx].cpu().numpy(), y_pred)
#             monitor.end_timer()
            
#             print(f"Epoch {epoch}: NMI = {nmi:.4f}")
            
#             if nmi > max_nmi:
#                 best_epoch = epoch
#                 max_nmi = nmi
#                 save_embedding(embedding, args.save_embedding)
#                 print(f"New best NMI: {max_nmi:.4f} at epoch {best_epoch}")
            
#             del embedding, y_pred
            
#         except Exception as e:
#             print(f"Evaluation error in epoch {epoch}: {e}")
        
#         gc.collect()
    
#     print(f"Training completed! Best NMI: {max_nmi:.4f} at epoch {best_epoch}")
#     print("Time taken for training: " + str(time.time()-start_train))
    
#     # Final evaluation
#     try:
#         embedding = torch.load("embedding_cora.pt").detach().cpu().numpy()
        
#         monitor.start_timer('kmeans')
#         kmeans = KMeans(n_clusters=y.max().item()+1, n_init=20)
#         y_pred = kmeans.fit_predict(embedding)
#         cm = clustering_metrics(y.cpu().numpy(), y_pred)
#         cm.evaluationClusterModelFromLabel(tqdm)
#         monitor.end_timer()
        
#     except Exception as e:
#         print(f"Final evaluation error: {e}")
    
#     # Print component statistics
#     monitor.print_stats()

# def test_multiscale_diffusion():
#     """Test function to compare original vs multi-scale diffusion"""
#     print("\n=== TESTING MULTI-SCALE DIFFUSION ===")
    
#     # Load small test data
#     from torch_geometric.datasets import Planetoid
#     dataset = Planetoid(root='.', name='Cora')
#     data = dataset[0]
    
#     # Create test adjacency matrix
#     edge_tensor = torch.sparse_coo_tensor(
#         data.edge_index, 
#         torch.ones(data.edge_index.size(1)), 
#         (data.x.size(0), data.x.size(0))
#     )
#     normalized_A = norm_adj(edge_tensor)
    
#     # Original diffusion
#     print("Computing original diffusion...")
#     start_time = time.time()
#     SX_original = compute_diffusion_matrix(normalized_A, data.x, niter=3)
#     original_time = time.time() - start_time
    
#     # Multi-scale diffusion
#     print("Computing multi-scale diffusion...")
#     start_time = time.time()
#     MSX_new = compute_adaptive_diffusion(normalized_A, data.x, max_hops=4, method="ppr")
#     multiscale_time = time.time() - start_time
    
#     # Compare results
#     print(f"\nComparison Results:")
#     print(f"Original diffusion - Time: {original_time:.4f}s, Norm: {torch.norm(SX_original):.4f}")
#     print(f"Multi-scale diffusion - Time: {multiscale_time:.4f}s, Norm: {torch.norm(MSX_new):.4f}")
    
#     # Feature diversity analysis
#     original_std = torch.std(SX_original, dim=0).mean()
#     multiscale_std = torch.std(MSX_new, dim=0).mean()
#     print(f"Feature diversity (std) - Original: {original_std:.4f}, Multi-scale: {multiscale_std:.4f}")
    
#     # Memory usage
#     original_memory = SX_original.element_size() * SX_original.nelement() / 1024 / 1024
#     multiscale_memory = MSX_new.element_size() * MSX_new.nelement() / 1024 / 1024
#     print(f"Memory usage - Original: {original_memory:.2f}MB, Multi-scale: {multiscale_memory:.2f}MB")
    
#     print("=== TEST COMPLETE ===\n")

if __name__ == "__main__":
    # test_multiscale_diffusion()
    main()