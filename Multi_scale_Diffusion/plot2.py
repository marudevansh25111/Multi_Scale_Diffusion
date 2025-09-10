import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Your data from the parameter study
cora_data = {
    'alpha': [0.15, 0.15, 0.15, 0.15, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2, 
              0.25, 0.25, 0.25, 0.25, 0.25, 0.3, 0.3, 0.3, 0.3, 0.3],
    't': [1, 3, 5, 7, 10, 1, 3, 5, 7, 10, 1, 3, 5, 7, 10, 1, 3, 5, 7, 10],
    'nmi_ppr': [0.5956, 0.5882, 0.5774, 0.5694, 0.5810, 0.5878, 0.5817, 0.5789, 
                0.5916, 0.5867, 0.5856, 0.5830, 0.5967, 0.5778, 0.5654, 0.5872, 
                0.5865, 0.5872, 0.5763, 0.5793],
    'nmi_heat': [0.6016, 0.5825, 0.5830, 0.5676, 0.5718, 0.5875, 0.5770, 0.5729, 
                 0.5929, 0.5839, 0.5776, 0.5762, 0.5929, 0.5820, 0.5839, 0.5844, 
                 0.5823, 0.5890, 0.5692, 0.5708]
}

citeseer_data = {
    'alpha': [0.15, 0.15, 0.15, 0.15, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2, 
              0.25, 0.25, 0.25, 0.25, 0.25, 0.3, 0.3, 0.3, 0.3, 0.3],
    't': [1, 3, 5, 7, 10, 1, 3, 5, 7, 10, 1, 3, 5, 7, 10, 1, 3, 5, 7, 10],
    'nmi_ppr': [0.3258, 0.3299, 0.3270, 0.3283, 0.3317, 0.3442, 0.3301, 0.3166, 
                0.3254, 0.3326, 0.3113, 0.3254, 0.3340, 0.3144, 0.3329, 0.3305, 
                0.3347, 0.3348, 0.3323, 0.3307],
    'nmi_heat': [0.3192, 0.3248, 0.3287, 0.3191, 0.3184, 0.3220, 0.3294, 0.3587, 
                 0.3250, 0.3329, 0.3113, 0.3229, 0.3176, 0.3139, 0.3305, 0.3233, 
                 0.3330, 0.3355, 0.3333, 0.3300]
}

pubmed_data = {
    'alpha': [0.15, 0.15, 0.15, 0.15, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2, 
              0.25, 0.25, 0.25, 0.25, 0.25, 0.3, 0.3, 0.3, 0.3, 0.3],
    't': [1, 3, 5, 7, 10, 1, 3, 5, 7, 10, 1, 3, 5, 7, 10, 1, 3, 5, 7, 10],
    'nmi_ppr': [0.3067, 0.3074, 0.3075, 0.3089, 0.3083, 0.3105, 0.3068, 0.3091, 
                0.3130, 0.3057, 0.3086, 0.3075, 0.3102, 0.3132, 0.3119, 0.3096, 
                0.3046, 0.3061, 0.3072, 0.3076],
    'nmi_heat': [0.3035, 0.2888, 0.2744, 0.3047, 0.3041, 0.3072, 0.3029, 0.3045, 
                 0.3078, 0.3037, 0.3043, 0.3040, 0.3065, 0.3097, 0.3075, 0.3057, 
                 0.2692, 0.2728, 0.2734, 0.3045]
}

def create_3d_surface_plot(data, dataset_name, method_name, nmi_key):
    """Create a 3D surface plot like the reference images"""
    
    # Unique values for alpha and t
    alpha_unique = sorted(list(set(data['alpha'])))
    t_unique = sorted(list(set(data['t'])))
    
    # Create meshgrid
    Alpha, T = np.meshgrid(alpha_unique, t_unique)
    
    # Create NMI matrix
    NMI = np.zeros((len(t_unique), len(alpha_unique)))
    
    for i, alpha_val in enumerate(alpha_unique):
        for j, t_val in enumerate(t_unique):
            # Find the corresponding NMI value
            for k in range(len(data['alpha'])):
                if data['alpha'][k] == alpha_val and data['t'][k] == t_val:
                    NMI[j, i] = data[nmi_key][k]
                    break
    
    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the surface plot
    surf = ax.plot_surface(Alpha, T, NMI, cmap=cm.viridis, 
                          alpha=0.8, linewidth=0.5, edgecolors='black')
    
    # Customize the plot
    ax.set_xlabel('Alpha', fontsize=12, labelpad=10)
    ax.set_ylabel('t', fontsize=12, labelpad=10)
    ax.set_zlabel('NMI', fontsize=12, labelpad=10)
    ax.set_title(f'{dataset_name} - {method_name}', fontsize=14, pad=20)
    
    # Add colorbar
    fig.colorbar(surf, shrink=0.5, aspect=20, label='NMI')
    
    # Set viewing angle (similar to reference images)
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    return fig

def create_all_plots():
    """Create all 6 plots and save them"""
    
    datasets = {
        'Cora': cora_data,
        'CiteSeer': citeseer_data,
        'PubMed': pubmed_data
    }
    
    methods = {
        'PPR': 'nmi_ppr',
        'Heat': 'nmi_heat'
    }
    
    # Create individual plots
    for dataset_name, data in datasets.items():
        for method_name, nmi_key in methods.items():
            fig = create_3d_surface_plot(data, dataset_name, method_name, nmi_key)
            plt.savefig(f'{dataset_name}_{method_name}_3D.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # Create a combined figure with all 6 subplots
    fig = plt.figure(figsize=(18, 12))
    
    plot_idx = 1
    for dataset_name, data in datasets.items():
        for method_name, nmi_key in methods.items():
            ax = fig.add_subplot(2, 3, plot_idx, projection='3d')
            
            # Prepare data
            alpha_unique = sorted(list(set(data['alpha'])))
            t_unique = sorted(list(set(data['t'])))
            Alpha, T = np.meshgrid(alpha_unique, t_unique)
            
            NMI = np.zeros((len(t_unique), len(alpha_unique)))
            for i, alpha_val in enumerate(alpha_unique):
                for j, t_val in enumerate(t_unique):
                    for k in range(len(data['alpha'])):
                        if data['alpha'][k] == alpha_val and data['t'][k] == t_val:
                            NMI[j, i] = data[nmi_key][k]
                            break
            
            # Create surface
            surf = ax.plot_surface(Alpha, T, NMI, cmap=cm.viridis, 
                                  alpha=0.8, linewidth=0.3, edgecolors='black')
            
            ax.set_xlabel('Alpha', fontsize=10)
            ax.set_ylabel('t', fontsize=10)
            ax.set_zlabel('NMI', fontsize=10)
            ax.set_title(f'{dataset_name} - {method_name}', fontsize=12)
            ax.view_init(elev=20, azim=45)
            
            plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('All_Parameter_Study_3D.png', dpi=300, bbox_inches='tight')
    plt.show()

# Alternative function for individual dataset plotting
def plot_single_dataset(dataset_name):
    """Plot both PPR and Heat for a single dataset side by side"""
    
    datasets = {
        'Cora': cora_data,
        'CiteSeer': citeseer_data,
        'PubMed': pubmed_data
    }
    
    data = datasets[dataset_name]
    
    fig = plt.figure(figsize=(15, 6))
    
    methods = [('PPR', 'nmi_ppr'), ('Heat', 'nmi_heat')]
    
    for i, (method_name, nmi_key) in enumerate(methods):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        
        # Prepare data
        alpha_unique = sorted(list(set(data['alpha'])))
        t_unique = sorted(list(set(data['t'])))
        Alpha, T = np.meshgrid(alpha_unique, t_unique)
        
        NMI = np.zeros((len(t_unique), len(alpha_unique)))
        for j, alpha_val in enumerate(alpha_unique):
            for k, t_val in enumerate(t_unique):
                for l in range(len(data['alpha'])):
                    if data['alpha'][l] == alpha_val and data['t'][l] == t_val:
                        NMI[k, j] = data[nmi_key][l]
                        break
        
        # Create surface
        surf = ax.plot_surface(Alpha, T, NMI, cmap=cm.viridis, 
                              alpha=0.8, linewidth=0.5, edgecolors='black')
        
        ax.set_xlabel('Alpha', fontsize=12)
        ax.set_ylabel('t', fontsize=12)
        ax.set_zlabel('NMI', fontsize=12)
        ax.set_title(f'{dataset_name} - {method_name}', fontsize=14)
        ax.view_init(elev=20, azim=45)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_comparison_3D.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Create all ultra-smooth plots
    print("Creating all 6 ultra-smooth surface plots...")
    create_all_plots()
    
    # Or create individual dataset comparisons
    # plot_single_dataset('Cora')
    # plot_single_dataset('CiteSeer')
    # plot_single_dataset('PubMed')
    
    print("All ultra-smooth plots saved successfully!")
    print("Required packages: pip install matplotlib numpy scipy")