# ─────────────────────────────────────────────
# 10. CARTE DES ROUTES
# ─────────────────────────────────────────────

from data import DEPOT
from utilities import route_distance
import matplotlib.pyplot as plt

VEHICLE_COLORS = [
    "#534AB7", "#1D9E75", "#D85A30", "#BA7517",
    "#E24B4A", "#185FA5", "#639922", "#D4537E",
]


def plot_routes_map(instance_id, routes, customers, p_best):
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_aspect("equal")
    ax.set_title(f"Instance {instance_id} — Routes QAOA (p={p_best})",
                 fontsize=13, pad=12)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    all_x  = [xy[0] for xy in customers.values()] + [0]
    all_y  = [xy[1] for xy in customers.values()] + [0]
    margin = 2.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.grid(True, color="#e0e0e0", linewidth=0.5, zorder=0)
    ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#cccccc", linewidth=0.8, zorder=1)

    import matplotlib.patches as mpatches
    legend_handles = []

    for v_idx, route in enumerate(routes):
        if not route:
            continue
        color  = VEHICLE_COLORS[v_idx % len(VEHICLE_COLORS)]
        dist   = route_distance(route, customers)
        full   = [None] + route + [None]

        def get_xy(cid):
            return customers[cid] if cid is not None else DEPOT

        for seg in range(len(full) - 1):
            x1, y1 = get_xy(full[seg])
            x2, y2 = get_xy(full[seg + 1])
            rad = 0.08 * (v_idx % 3 - 1)
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6,
                                        connectionstyle=f"arc3,rad={rad}",
                                        mutation_scale=14),
                        zorder=3)

        legend_handles.append(mpatches.Patch(
            color=color,
            label=f"V{v_idx+1} : {route}  (dist={dist:.2f})"
        ))

    client_color = {}
    for v_idx, route in enumerate(routes):
        for cid in route:
            client_color[cid] = VEHICLE_COLORS[v_idx % len(VEHICLE_COLORS)]

    for cid, (cx, cy) in customers.items():
        col = client_color.get(cid, "#888888")
        ax.plot(cx, cy, "o", color=col, markersize=18,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        ax.text(cx, cy, str(cid), ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=6)

    ax.plot(0, 0, "*", color="black", markersize=20,
            markeredgecolor="white", markeredgewidth=1, zorder=7)
    ax.text(0.3, 0.3, "Dépôt", fontsize=8, color="#333333", zorder=7)

    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(0.0, -0.02), ncol=2, fontsize=8,
              frameon=True, framealpha=0.9, edgecolor="#cccccc")

    plt.tight_layout()
    path = f"img/map_instance{instance_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Carte des routes : {path}")