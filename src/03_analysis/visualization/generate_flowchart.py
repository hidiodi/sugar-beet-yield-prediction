
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Define Box Styles
    box_style = dict(boxstyle="round,pad=0.3", ec="black", lw=1.5)

    # --- LEVEL 1: DATA SOURCES ---
    ax.text(2, 9, "Data Sources", fontsize=12, fontweight='bold', ha='center')

    # Weather
    ax.text(2, 8.2, "Weather Grids\n(HYRAS 2.0)",
            bbox=dict(fc="#e1f5fe", **box_style), ha='center', va='center')
    # Soil
    ax.text(4.5, 8.2, "Soil Props\n(SoilGrids)",
            bbox=dict(fc="#e1f5fe", **box_style), ha='center', va='center')
    # Remote Sensing
    ax.text(7, 8.2, "Remote Sensing\n(MODIS)",
            bbox=dict(fc="#e1f5fe", **box_style), ha='center', va='center')

    # --- LEVEL 2: FEATURE ENGINEERING ---
    ax.text(4.5, 7, "Feature Engineering", fontsize=12, fontweight='bold', ha='center')

    # WOFOST Simulation
    ax.text(3, 6, "WOFOST 7.2\nSimulation\n(Bio-Physical Scanner)",
            bbox=dict(fc="#fff9c4", **box_style), ha='center', va='center')

    # Z-Scores
    ax.text(6, 6, "Statistical\nZ-Scores\n(Expanding Window)",
            bbox=dict(fc="#fff9c4", **box_style), ha='center', va='center')

    # Derived Indices
    ax.text(4.5, 4.8, "Compound Indices\n(Scorch, Drown, Bumper)",
            bbox=dict(fc="#ffe0b2", **box_style), ha='center', va='center')

    # --- LEVEL 3: COMPONENT MODELS ---
    ax.text(11, 9, "Component Models", fontsize=12, fontweight='bold', ha='center')

    # Trend
    ax.text(11, 8.2, "1. Statistical Trend\n(GAM + ARIMA)",
            bbox=dict(fc="#c8e6c9", **box_style), ha='center', va='center')
    # Native
    ax.text(11, 7.2, "2. Native Ensemble\n(Quantile XGB)",
            bbox=dict(fc="#c8e6c9", **box_style), ha='center', va='center')
    # Hybrid
    ax.text(11, 6.2, "3. Hybrid XGBoost\n(Yield Ratio)",
            bbox=dict(fc="#c8e6c9", **box_style), ha='center', va='center')
    # Robust
    ax.text(11, 5.2, "4. Robust Linear\n(Huber Regressor)",
            bbox=dict(fc="#c8e6c9", **box_style), ha='center', va='center')
    # Heat Signal
    ax.text(11, 4.2, "5. Multivariate Heat\n(Extreme Weighted)",
            bbox=dict(fc="#c8e6c9", **box_style), ha='center', va='center')

    # --- LEVEL 4: META LEARNER ---
    ax.text(7, 3, "Meta-Learner Architecture", fontsize=12, fontweight='bold', ha='center')

    # Input Features for Meta
    ax.text(4, 2.5, "Meta-Features:\n- Ensemble Variance\n- Hist. Bias\n- Bio-Physical Context",
            bbox=dict(fc="#d1c4e9", **box_style), ha='center', va='center')

    # The Classifier
    ax.text(7, 2.5, "Meta-Learner\n(XGBoost Classifier)",
            bbox=dict(fc="#9575cd", ec="black", lw=2, boxstyle="round,pad=0.5", alpha=0.8),
            color="white", fontweight='bold', ha='center', va='center')

    # Output Probabilities
    ax.text(10, 2.5, "Regime Probabilities\n(Soft Voting Weights)",
            bbox=dict(fc="#d1c4e9", **box_style), ha='center', va='center')

    # --- LEVEL 5: FINAL OUTPUT ---
    ax.text(7, 0.8, "Final Forecast\n(Weighted Average)",
            bbox=dict(fc="#ffcc80", ec="black", lw=2, boxstyle="round,pad=0.5"),
            fontsize=12, fontweight='bold', ha='center', va='center')

    # --- ARROWS ---

    # Data -> WOFOST
    ax.annotate("", xy=(3, 6.5), xytext=(2, 7.8), arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("", xy=(3, 6.5), xytext=(4.5, 7.8), arrowprops=dict(arrowstyle="->", lw=1))

    # Data -> Z-Scores
    ax.annotate("", xy=(6, 6.5), xytext=(2, 7.8), arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("", xy=(6, 6.5), xytext=(7, 7.8), arrowprops=dict(arrowstyle="->", lw=1))

    # WOFOST/Z -> Indices
    ax.annotate("", xy=(4.5, 5.3), xytext=(3, 5.6), arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("", xy=(4.5, 5.3), xytext=(6, 5.6), arrowprops=dict(arrowstyle="->", lw=1))

    # Indices -> Component Models (Conceptual Fan Out)
    # Just draw a line from Indices to a mid point then to models?
    # Or just imply it. Let's draw arrows from Indices to the "Component Models" block area
    ax.annotate("", xy=(9.5, 6.5), xytext=(6, 4.8), arrowprops=dict(arrowstyle="->", lw=1))

    # Component Models -> Meta Learner (Predictions)
    ax.annotate("", xy=(7, 3), xytext=(11, 3.8), arrowprops=dict(arrowstyle="->", lw=1, connectionstyle="arc3,rad=-0.2"))

    # Meta Features -> Meta Learner
    ax.annotate("", xy=(5.8, 2.5), xytext=(5.3, 2.5), arrowprops=dict(arrowstyle="->", lw=1))

    # Meta Learner -> Probabilities
    ax.annotate("", xy=(8.5, 2.5), xytext=(8.2, 2.5), arrowprops=dict(arrowstyle="->", lw=1))

    # Probabilities -> Final
    ax.annotate("", xy=(7, 1.3), xytext=(10, 2.1), arrowprops=dict(arrowstyle="->", lw=1))

    # Component Models -> Final (The actual values)
    ax.annotate("", xy=(7, 1.3), xytext=(11, 3.8), arrowprops=dict(arrowstyle="->", lw=1, connectionstyle="arc3,rad=0.3"))

    plt.tight_layout()
    plt.savefig('docs/paper_latex/figures/super_ensemble_flowchart.png', dpi=300, bbox_inches='tight')
    print("Flowchart saved.")

if __name__ == "__main__":
    draw_flowchart()
