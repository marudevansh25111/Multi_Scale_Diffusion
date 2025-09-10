import matplotlib.pyplot as plt
import numpy as np

# Set style for research paper
plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['grid.alpha'] = 0.3

# Data from the parameter study document
# Alpha Parameter Study Results (t=5)
alpha_values = [0.15, 0.20, 0.25, 0.30]
alpha_data = {
    'Cora': [0.6016, 0.5882, 0.5830, 0.5898],
    'CiteSeer': [0.3258, 0.3299, 0.3289, 0.3311], 
    'PubMed': [0.3067, 0.3074, 0.3075, 0.3089]
}

# T Parameter Study Results (alpha=0.2)
t_values = [1, 3, 5, 7, 10]
t_data = {
    'Cora': [0.5810, 0.5911, 0.5817, 0.5789, 0.5929],
    'CiteSeer': [0.3317, 0.3442, 0.3301, 0.3172, 0.3258],
    'PubMed': [0.3083, 0.3105, 0.3068, 0.3091, 0.3130]
}

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Colors for each dataset (grayscale for print compatibility)
colors = ['#2C3E50', '#7F8C8D', '#BDC3C7']
markers = ['o', 's', '^']
linestyles = ['-', '--', '-.']

# Plot 1: Alpha Parameter Study
ax1.set_title('(a) Alpha Parameter Study (t=5)', fontweight='bold', pad=20)
for i, (dataset, values) in enumerate(alpha_data.items()):
    ax1.plot(alpha_values, values, 
             color=colors[i], 
             marker=markers[i], 
             linestyle=linestyles[i],
             linewidth=2, 
             markersize=8, 
             label=dataset,
             markerfacecolor='white',
             markeredgewidth=2)

ax1.set_xlabel('Alpha Parameter (α)', fontweight='bold')
ax1.set_ylabel('NMI Score', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(frameon=True, fancybox=False, shadow=False)
ax1.set_xlim(0.14, 0.31)
ax1.set_ylim(0.30, 0.62)

# Add value labels on points for Alpha graph
for i, (dataset, values) in enumerate(alpha_data.items()):
    for j, (x, y) in enumerate(zip(alpha_values, values)):
        ax1.annotate(f'{y:.3f}', 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=9)

# Plot 2: T Parameter Study
ax2.set_title('(b) T Parameter Study (α=0.2)', fontweight='bold', pad=20)
for i, (dataset, values) in enumerate(t_data.items()):
    ax2.plot(t_values, values, 
             color=colors[i], 
             marker=markers[i], 
             linestyle=linestyles[i],
             linewidth=2, 
             markersize=8, 
             label=dataset,
             markerfacecolor='white',
             markeredgewidth=2)

ax2.set_xlabel('T Parameter (t)', fontweight='bold')
ax2.set_ylabel('NMI Score', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(frameon=True, fancybox=False, shadow=False)
ax2.set_xlim(0.5, 10.5)
ax2.set_ylim(0.30, 0.62)

# Add value labels on points for T graph
for i, (dataset, values) in enumerate(t_data.items()):
    for j, (x, y) in enumerate(zip(t_values, values)):
        ax2.annotate(f'{y:.3f}', 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=9)

# Adjust layout and save
plt.tight_layout()
plt.savefig('parameter_study_results.png', dpi=300, bbox_inches='tight')
plt.savefig('parameter_study_results.pdf', bbox_inches='tight')
plt.show()

# Print summary statistics
print("=" * 60)
print("PARAMETER STUDY SUMMARY")
print("=" * 60)

print("\nAlpha Parameter Study (t=5):")
print("-" * 30)
for dataset in alpha_data.keys():
    best_alpha_idx = np.argmax(alpha_data[dataset])
    best_alpha = alpha_values[best_alpha_idx]
    best_nmi = alpha_data[dataset][best_alpha_idx]
    print(f"{dataset:>10}: Best α={best_alpha}, NMI={best_nmi:.4f}")

print("\nT Parameter Study (α=0.2):")
print("-" * 30)
for dataset in t_data.keys():
    best_t_idx = np.argmax(t_data[dataset])
    best_t = t_values[best_t_idx]
    best_nmi = t_data[dataset][best_t_idx]
    print(f"{dataset:>10}: Best t={best_t}, NMI={best_nmi:.4f}")

# Create individual bar charts for better visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart for Alpha parameter
x_pos = np.arange(len(alpha_values))
width = 0.25

for i, (dataset, values) in enumerate(alpha_data.items()):
    ax1.bar(x_pos + i*width, values, width, 
           label=dataset, color=colors[i], alpha=0.8,
           edgecolor='black', linewidth=0.5)

ax1.set_xlabel('Alpha Parameter (α)', fontweight='bold')
ax1.set_ylabel('NMI Score', fontweight='bold')
ax1.set_title('(a) Alpha Parameter Study (t=5)', fontweight='bold', pad=20)
ax1.set_xticks(x_pos + width)
ax1.set_xticklabels([f'{a:.2f}' for a in alpha_values])
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 0.65)

# Add value labels on bars for Alpha
for i, (dataset, values) in enumerate(alpha_data.items()):
    for j, v in enumerate(values):
        ax1.text(j + i*width, v + 0.01, f'{v:.3f}', 
                ha='center', va='bottom', fontsize=9, rotation=90)

# Bar chart for T parameter
x_pos2 = np.arange(len(t_values))

for i, (dataset, values) in enumerate(t_data.items()):
    ax2.bar(x_pos2 + i*width, values, width, 
           label=dataset, color=colors[i], alpha=0.8,
           edgecolor='black', linewidth=0.5)

ax2.set_xlabel('T Parameter (t)', fontweight='bold')
ax2.set_ylabel('NMI Score', fontweight='bold')
ax2.set_title('(b) T Parameter Study (α=0.2)', fontweight='bold', pad=20)
ax2.set_xticks(x_pos2 + width)
ax2.set_xticklabels([str(t) for t in t_values])
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 0.65)

# Add value labels on bars for T
for i, (dataset, values) in enumerate(t_data.items()):
    for j, v in enumerate(values):
        ax2.text(j + i*width, v + 0.01, f'{v:.3f}', 
                ha='center', va='bottom', fontsize=9, rotation=90)

plt.tight_layout()
plt.savefig('parameter_study_bars.png', dpi=300, bbox_inches='tight')
plt.savefig('parameter_study_bars.pdf', bbox_inches='tight')
plt.show()

# Create data table for reference
print("\n" + "=" * 60)
print("DETAILED RESULTS TABLE")
print("=" * 60)

print("\nAlpha Parameter Study (t=5):")
print("Alpha\tCora\tCiteSeer\tPubMed")
print("-" * 40)
for i, alpha in enumerate(alpha_values):
    print(f"{alpha:.2f}\t{alpha_data['Cora'][i]:.4f}\t{alpha_data['CiteSeer'][i]:.4f}\t\t{alpha_data['PubMed'][i]:.4f}")

print("\nT Parameter Study (α=0.2):")
print("T\tCora\tCiteSeer\tPubMed")
print("-" * 40)
for i, t in enumerate(t_values):
    print(f"{t}\t{t_data['Cora'][i]:.4f}\t{t_data['CiteSeer'][i]:.4f}\t\t{t_data['PubMed'][i]:.4f}")