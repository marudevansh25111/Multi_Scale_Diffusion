# =============================================================================
# SEPARATE GRAPH GENERATION FOR RESEARCH PAPER
# This code creates 4 separate graph files (each as PNG and PDF):
# 1. Alpha Parameter Study - Line Plot
# 2. T Parameter Study - Line Plot  
# 3. Alpha Parameter Study - Bar Chart
# 4. T Parameter Study - Bar Chart
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

# Set style for research paper
plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3

# Data from the parameter study document
print("GENERATING SEPARATE GRAPHS FOR PARAMETER STUDY")
print("=" * 50)
print("Files that will be generated:")
print("1. alpha_parameter_study.png/pdf (Line plot)")
print("2. t_parameter_study.png/pdf (Line plot)")
print("3. alpha_parameter_bars.png/pdf (Bar chart)")
print("4. t_parameter_bars.png/pdf (Bar chart)")
print("=" * 50)

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

# Colors for each dataset (grayscale for print compatibility)
colors = ['#2C3E50', '#7F8C8D', '#BDC3C7']
markers = ['o', 's', '^']
linestyles = ['-', '--', '-.']

# =============================================================================
# GRAPH 1: ALPHA PARAMETER STUDY (LINE PLOT)
# =============================================================================

fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))

ax1.set_title('Alpha Parameter Study (t=5)', fontweight='bold', pad=20, fontsize=16)
for i, (dataset, values) in enumerate(alpha_data.items()):
    ax1.plot(alpha_values, values, 
             color=colors[i], 
             marker=markers[i], 
             linestyle=linestyles[i],
             linewidth=3, 
             markersize=12, 
             label=dataset,
             markerfacecolor='white',
             markeredgewidth=2)

ax1.set_xlabel('Alpha Parameter (α)', fontweight='bold', fontsize=14)
ax1.set_ylabel('NMI Score', fontweight='bold', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(frameon=True, fancybox=False, shadow=False, fontsize=12)
ax1.set_xlim(0.14, 0.31)
ax1.set_ylim(0.30, 0.62)

# Add value labels on points for Alpha graph
for i, (dataset, values) in enumerate(alpha_data.items()):
    for j, (x, y) in enumerate(zip(alpha_values, values)):
        ax1.annotate(f'{y:.3f}', 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,15), 
                    ha='center',
                    fontsize=11,
                    fontweight='bold')

plt.tight_layout()
plt.savefig('alpha_parameter_study.png', dpi=300, bbox_inches='tight')
plt.savefig('alpha_parameter_study.pdf', bbox_inches='tight')
plt.show()
print("✓ Generated: alpha_parameter_study.png and alpha_parameter_study.pdf")

# =============================================================================
# GRAPH 2: T PARAMETER STUDY (LINE PLOT)
# =============================================================================

fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))

ax2.set_title('T Parameter Study (α=0.2)', fontweight='bold', pad=20, fontsize=16)
for i, (dataset, values) in enumerate(t_data.items()):
    ax2.plot(t_values, values, 
             color=colors[i], 
             marker=markers[i], 
             linestyle=linestyles[i],
             linewidth=3, 
             markersize=12, 
             label=dataset,
             markerfacecolor='white',
             markeredgewidth=2)

ax2.set_xlabel('T Parameter (t)', fontweight='bold', fontsize=14)
ax2.set_ylabel('NMI Score', fontweight='bold', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(frameon=True, fancybox=False, shadow=False, fontsize=12)
ax2.set_xlim(0.5, 10.5)
ax2.set_ylim(0.30, 0.62)

# Add value labels on points for T graph
for i, (dataset, values) in enumerate(t_data.items()):
    for j, (x, y) in enumerate(zip(t_values, values)):
        ax2.annotate(f'{y:.3f}', 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,15), 
                    ha='center',
                    fontsize=11,
                    fontweight='bold')

plt.tight_layout()
plt.savefig('t_parameter_study.png', dpi=300, bbox_inches='tight')
plt.savefig('t_parameter_study.pdf', bbox_inches='tight')
plt.show()
print("✓ Generated: t_parameter_study.png and t_parameter_study.pdf")

# =============================================================================
# GRAPH 3: ALPHA PARAMETER STUDY (BAR CHART)
# =============================================================================

fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))

x_pos = np.arange(len(alpha_values))
width = 0.25

for i, (dataset, values) in enumerate(alpha_data.items()):
    ax3.bar(x_pos + i*width, values, width, 
           label=dataset, color=colors[i], alpha=0.8,
           edgecolor='black', linewidth=0.5)

ax3.set_xlabel('Alpha Parameter (α)', fontweight='bold', fontsize=14)
ax3.set_ylabel('NMI Score', fontweight='bold', fontsize=14)
ax3.set_title('Alpha Parameter Study (t=5)', fontweight='bold', pad=20, fontsize=16)
ax3.set_xticks(x_pos + width)
ax3.set_xticklabels([f'{a:.2f}' for a in alpha_values])
ax3.legend(fontsize=12)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, 0.65)

# Add value labels on bars for Alpha
for i, (dataset, values) in enumerate(alpha_data.items()):
    for j, v in enumerate(values):
        ax3.text(j + i*width, v + 0.01, f'{v:.3f}', 
                ha='center', va='bottom', fontsize=11, rotation=90, fontweight='bold')

plt.tight_layout()
plt.savefig('alpha_parameter_bars.png', dpi=300, bbox_inches='tight')
plt.savefig('alpha_parameter_bars.pdf', bbox_inches='tight')
plt.show()
print("✓ Generated: alpha_parameter_bars.png and alpha_parameter_bars.pdf")

# =============================================================================
# GRAPH 4: T PARAMETER STUDY (BAR CHART)
# =============================================================================

fig4, ax4 = plt.subplots(1, 1, figsize=(10, 8))

x_pos2 = np.arange(len(t_values))

for i, (dataset, values) in enumerate(t_data.items()):
    ax4.bar(x_pos2 + i*width, values, width, 
           label=dataset, color=colors[i], alpha=0.8,
           edgecolor='black', linewidth=0.5)

ax4.set_xlabel('T Parameter (t)', fontweight='bold', fontsize=14)
ax4.set_ylabel('NMI Score', fontweight='bold', fontsize=14)
ax4.set_title('T Parameter Study (α=0.2)', fontweight='bold', pad=20, fontsize=16)
ax4.set_xticks(x_pos2 + width)
ax4.set_xticklabels([str(t) for t in t_values])
ax4.legend(fontsize=12)
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim(0, 0.65)

# Add value labels on bars for T
for i, (dataset, values) in enumerate(t_data.items()):
    for j, v in enumerate(values):
        ax4.text(j + i*width, v + 0.01, f'{v:.3f}', 
                ha='center', va='bottom', fontsize=11, rotation=90, fontweight='bold')

plt.tight_layout()
plt.savefig('t_parameter_bars.png', dpi=300, bbox_inches='tight')
plt.savefig('t_parameter_bars.pdf', bbox_inches='tight')
plt.show()
print("✓ Generated: t_parameter_bars.png and t_parameter_bars.pdf")

# =============================================================================
# SUMMARY AND DATA TABLES
# =============================================================================

print("\n" + "=" * 60)
print("PARAMETER STUDY SUMMARY - SEPARATE GRAPHS GENERATED")
print("=" * 60)
print("\nFiles Generated:")
print("- alpha_parameter_study.png/pdf (Alpha line plot)")
print("- t_parameter_study.png/pdf (T line plot)")
print("- alpha_parameter_bars.png/pdf (Alpha bar chart)")
print("- t_parameter_bars.png/pdf (T bar chart)")
print("\nBest Results:")

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

print("\n" + "=" * 60)
print("✓ ALL 8 FILES SAVED SUCCESSFULLY!")
print("You now have 4 separate graphs (each as PNG + PDF)")
print("Total: 8 files generated")
print("=" * 60)