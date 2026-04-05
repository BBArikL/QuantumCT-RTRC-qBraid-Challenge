# ─────────────────────────────────────────────
# 11. GRAPHIQUE DE CONVERGENCE
# ─────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt

def plot_convergence(instance_id, convergence_data, p_values, true_optimal=None):
    COLORS = {1: "#534AB7", 2: "#1D9E75", 3: "#D85A30"}

    has_ratio = true_optimal is not None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax1, ax2  = axes

    fig.suptitle(f"Instance {instance_id} — QAOA Convergence per Depth p",
                 fontsize=13, y=1.01)

    # Panneau gauche : convergence énergie Ising
    for p in p_values:
        hist = convergence_data[p].get("history", [])
        if not hist:
            continue
        best_so_far = np.minimum.accumulate(hist)
        ax1.plot(np.arange(1, len(best_so_far)+1), best_so_far,
                 color=COLORS.get(p, "gray"), linewidth=2,
                 label=f"p={p}  (dist={convergence_data[p]['distance']:.3f})")
        ax1.axhline(best_so_far[-1], color=COLORS.get(p, "gray"),
                    linestyle="--", linewidth=0.7, alpha=0.4)

    ax1.set_xlabel("Iterations COBYLA")
    ax1.set_ylabel("Best Ising Energy")
    ax1.set_title("Convergence")
    ax1.legend(fontsize=9)
    ax1.spines[["top","right"]].set_visible(False)
    ax1.grid(axis="y", linewidth=0.4, alpha=0.5)

    # Panneau droit : ratio d'approximation global par p
    if has_ratio:
        ps_r   = [p for p in p_values if convergence_data[p].get("ratio") is not None]
        ratios = [convergence_data[p]["ratio"] for p in ps_r]

        bars = ax2.bar([str(p) for p in ps_r], ratios,
                       color=[COLORS.get(p, "gray") for p in ps_r],
                       width=0.4, edgecolor="none")
        for bar, r in zip(bars, ratios):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.002,
                     f"{r:.4f}",
                     ha="center", va="bottom", fontsize=9, color="#2C2C2A")

        ax2.axhline(1.0, color="#E24B4A", linestyle="--",
                    linewidth=1.2, label=f"Optimal = {true_optimal:.4f}")
        ax2.axhline(1.05, color="#BA7517", linestyle=":",
                    linewidth=1.0, label="Seuil 5%")
        y_max = max(ratios) + 0.05
        y_min = max(0.90, min(ratios) - 0.05)
        ax2.set_ylim(y_min, y_max)
        ax2.set_xlabel("Depth p")
        ax2.set_ylabel("Ratio global (QAOA / optimal)")
        ax2.set_title("Global Approximation Ratio")
        ax2.legend(fontsize=9)
    else:
        # Pas d'optimal disponible : afficher les distances absolues
        ps_v   = list(p_values)
        dists  = [convergence_data[p]["distance"] for p in ps_v]
        bars   = ax2.bar([str(p) for p in ps_v], dists,
                         color=[COLORS.get(p, "gray") for p in ps_v],
                         width=0.4, edgecolor="none")
        for bar, d in zip(bars, dists):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.01 * max(dists),
                     f"{d:.3f}",
                     ha="center", va="bottom", fontsize=9, color="#2C2C2A")
        ax2.set_xlabel("Depth p")
        ax2.set_ylabel("Distance totale QAOA")
        ax2.set_title("Solution Quality (no exact baseline)")

    ax2.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    path = f"img/convergence_instance{instance_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Graphique : {path}")
