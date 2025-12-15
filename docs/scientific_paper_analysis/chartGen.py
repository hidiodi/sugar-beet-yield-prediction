import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch


def draw_flowchart():
    """Generates a simple flowchart of the Super Ensemble System Architecture using matplotlib."""

    # 1. Setup the figure
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    # Define node properties
    node_width = 2.5
    node_height = 0.5
    box_style = "round,pad=0.2"

    # Define Node Coordinates (Y_center adjusted down to make space at the top)
    coords = {
        'RawData': (2.5, 6.0),  # New top node
        'Data': (2.5, 5.0),
        'Features': (2.5, 4.0),
        'Components': (2.5, 2.7),
        'MetaLearner': (2.5, 1.4),
        'Output': (2.5, 0.4)
    }

    # --- 2. Draw Main Processing Nodes ---

    # Node 0: Raw Data Sources (New Top Node)
    ax.add_patch(FancyBboxPatch(
        (coords['RawData'][0] - node_width / 2, coords['RawData'][1] - node_height / 2),
        node_width, node_height,
        boxstyle=box_style, facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2,
    ))
    plt.text(coords['RawData'][0], coords['RawData'][1],
             '0. Raw Data Sources\n(Yield, Weather, SoilGrids, NDVI)',
             ha='center', va='center', fontsize=10, weight='bold')

    # Node 1: Data & WOFOST
    ax.add_patch(FancyBboxPatch(
        (coords['Data'][0] - node_width / 2, coords['Data'][1] - node_height / 2),
        node_width, node_height,
        boxstyle=box_style, facecolor='#E0F7FA', edgecolor='#00796B', linewidth=2
    ))
    plt.text(coords['Data'][0], coords['Data'][1],
             '1. Data Preprocessing & WOFOST 7.2\n(Physics Simulation, Heat Signal Proxy)',
             ha='center', va='center', fontsize=10, weight='bold')

    # Node 2: Feature Engineering
    ax.add_patch(FancyBboxPatch(
        (coords['Features'][0] - node_width / 2, coords['Features'][1] - node_height / 2),
        node_width, node_height,
        boxstyle=box_style, facecolor='#B3E5FC', edgecolor='#0288D1', linewidth=2
    ))
    plt.text(coords['Features'][0], coords['Features'][1],
             '2. Feature Engineering Pipeline\n(Z-Scores, Scorch Index, Proxies)',
             ha='center', va='center', fontsize=10, weight='bold')

    # Node 4: Super Ensemble (Meta-Learner & Soft Voting)
    ax.add_patch(FancyBboxPatch(
        (coords['MetaLearner'][0] - node_width / 2, coords['MetaLearner'][1] - node_height / 2),
        node_width, node_height,
        boxstyle=box_style, facecolor='#FFAB91', edgecolor='#E64A19', linewidth=2
    ))
    plt.text(coords['MetaLearner'][0], coords['MetaLearner'][1],
             '4. Super Ensemble (Meta-Learner)\n& Soft Voting Inference',
             ha='center', va='center', fontsize=10, weight='bold')

    # Node 5: Final Output
    ax.add_patch(FancyBboxPatch(
        (coords['Output'][0] - node_width / 2, coords['Output'][1] - node_height / 2),
        node_width, node_height,
        boxstyle=box_style, facecolor='#4CAF50', edgecolor='#388E3C', linewidth=2
    ))
    plt.text(coords['Output'][0], coords['Output'][1],
             '5. Final Yield Forecast',
             ha='center', va='center', fontsize=10, weight='bold', color='white')

    # --- 3. Draw Component Models Group (Cluster) ---
    cluster_x = 1.0
    cluster_y = 2.0
    cluster_width = 3.0
    cluster_height = 1.4

    # Group box (dashed rectangle)
    ax.add_patch(patches.Rectangle(
        (cluster_x, cluster_y), cluster_width, cluster_height,
        facecolor='none', edgecolor='#8E24AA', linestyle='--', linewidth=1.5,
        zorder=0
    ))
    plt.text(cluster_x + 0.05, cluster_y + cluster_height - 0.1,
             '3. Component Models', ha='left', va='top', fontsize=10, color='#6A1B9A')

    # Internal component nodes (smaller boxes)
    comp_node_width = 0.8
    comp_node_height = 0.4

    comp_coords = {
        'T': (1.4, 2.9),  # Trend
        'P': (2.4, 2.9),  # Physics
        'H': (3.4, 2.9),  # Hybrid
        'R': (2.4, 2.3)  # Robust
    }

    # Trend
    ax.add_patch(patches.Rectangle(
        (comp_coords['T'][0] - comp_node_width / 2, comp_coords['T'][1] - comp_node_height / 2),
        comp_node_width, comp_node_height,
        facecolor='#F3E5F5', edgecolor='#6A1B9A', linewidth=1
    ))
    plt.text(comp_coords['T'][0], comp_coords['T'][1], 'Trend\n(GAM+ARIMA)', ha='center', va='center', fontsize=8)

    # Physics
    ax.add_patch(patches.Rectangle(
        (comp_coords['P'][0] - comp_node_width / 2, comp_coords['P'][1] - comp_node_height / 2),
        comp_node_width, comp_node_height,
        facecolor='#F3E5F5', edgecolor='#6A1B9A', linewidth=1
    ))
    plt.text(comp_coords['P'][0], comp_coords['P'][1], 'Physics\n(Quantile)', ha='center', va='center', fontsize=8)

    # Hybrid
    ax.add_patch(patches.Rectangle(
        (comp_coords['H'][0] - comp_node_width / 2, comp_coords['H'][1] - comp_node_height / 2),
        comp_node_width, comp_node_height,
        facecolor='#F3E5F5', edgecolor='#6A1B9A', linewidth=1
    ))
    plt.text(comp_coords['H'][0], comp_coords['H'][1], 'Hybrid\n(XGB Ratio)', ha='center', va='center', fontsize=8)

    # Robust
    ax.add_patch(patches.Rectangle(
        (comp_coords['R'][0] - comp_node_width / 2, comp_coords['R'][1] - comp_node_height / 2),
        comp_node_width, comp_node_height,
        facecolor='#F3E5F5', edgecolor='#6A1B9A', linewidth=1
    ))
    plt.text(comp_coords['R'][0], comp_coords['R'][1], 'Robust\n(Huber)', ha='center', va='center', fontsize=8)

    # --- 4. Draw Arrows for Flow ---

    # Raw Data -> Data Prep
    plt.arrow(coords['RawData'][0], coords['RawData'][1] - node_height / 2, 0, -0.2,
              head_width=0.1, head_length=0.1, fc='black', ec='black', length_includes_head=True)

    # Data Prep -> Features
    plt.arrow(coords['Data'][0], coords['Data'][1] - node_height / 2, 0, -0.2,
              head_width=0.1, head_length=0.1, fc='black', ec='black', length_includes_head=True)

    # Features -> Components
    plt.arrow(coords['Features'][0], coords['Features'][1] - node_height / 2, 0, -0.3,
              head_width=0.1, head_length=0.1, fc='black', ec='black', length_includes_head=True)

    # Components -> Meta-Learner (via side route for cluster)
    arrow_start = coords['Components'][1] - cluster_height / 2 + 0.1  # just below the component group
    arrow_end = coords['MetaLearner'][1] + node_height / 2 - 0.6
    plt.arrow(coords['Components'][0], arrow_start, 0, arrow_end - arrow_start,
              head_width=0.1, head_length=0.1, fc='black', ec='black', length_includes_head=True)
    plt.text(coords['Components'][0] + 0.1, arrow_start - 0.2,
             'Predictions\n& Features', ha='left', va='center', fontsize=9)

    # Meta-Learner -> Output
    plt.arrow(coords['MetaLearner'][0], coords['MetaLearner'][1] - node_height / 2, 0, -0.5,
              head_width=0.1, head_length=0.1, fc='black', ec='black', length_includes_head=True)

    # Title
    fig.suptitle('Super Ensemble Yield Forecasting System Architecture', fontsize=14, weight='bold')

    plt.savefig('super_ensemble_flowchart.png')
    plt.close()


draw_flowchart()