import matplotlib.pyplot as plt

# One colour per group, used in every chart, notebook and app page
COLORS = {"Non-CKD": "#2a78d6", "CKD": "#eb6834"}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "legend.frameon": False,
    })


def grouped_bar(ax, table, title, xlabel, ylabel="% of patients in group"):
    """table: rows = categories (in order), columns = 'Non-CKD' and 'CKD' (percentages)."""
    groups = [g for g in ["Non-CKD", "CKD"] if g in table.columns]
    n = len(groups)
    width = 0.38
    x = range(len(table.index))
    for i, g in enumerate(groups):
        offset = (i - (n - 1) / 2) * (width + 0.03)
        ax.bar([xi + offset for xi in x], table[g], width=width,
               color=COLORS[g], label=g, edgecolor="white", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(i) for i in table.index], rotation=35, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    return ax


def grouped_barh(ax, table, title, xlabel="% of patients in group"):
    """Horizontal grouped bars. table: rows = items (top to bottom), columns = 'Non-CKD' and 'CKD'."""
    groups = [g for g in ["Non-CKD", "CKD"] if g in table.columns]
    height = 0.36
    y = list(range(len(table.index)))[::-1]
    for i, g in enumerate(groups):
        offset = (i - (len(groups) - 1) / 2) * (height + 0.04)
        vals = table[g].values
        bars = ax.barh([yi - offset for yi in y], vals, height=height,
                       color=COLORS[g], label=g, edgecolor="white", linewidth=0.8)
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{v:.0f}%",
                    va="center", fontsize=8, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(table.index)
    ax.set_xlim(0, 105)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend(loc="lower right")
    return ax


def _blend(hex_color, strength):
    """Mix a colour with white: strength 0 = white, 1 = full colour."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(hex_color)
    return (1 - strength * (1 - r), 1 - strength * (1 - g), 1 - strength * (1 - b))


def readable_split(feature, threshold, raw=None):
    """Turn a coded split such as 'hemo <= 5.5' into a readable rule such as '<= 12.6'."""
    import math
    k = math.floor(threshold)
    if raw is not None and feature in raw.columns:
        values = raw[feature].dropna().unique()
        if not set(values) <= {"0", "1", "2"}:
            from data_prep import _bin_key
            label = sorted(values, key=_bin_key)[k]
            if label == "< 0":
                return "= 0"
            if " - " in label:
                return "≤ " + label.split(" - ")[1]
            return label
    if set([threshold]) <= {0.5} and feature not in ("bp limit",):
        return "= No"
    return f"≤ {k}"


def plot_readable_tree(ax, tree, features, raw=None, names=None, fontsize=9):
    """Draw a decision tree with readable rules and the project colours (CKD orange, non-CKD blue)."""
    import re
    from sklearn.tree import plot_tree
    names = names or {}
    anns = plot_tree(tree, feature_names=features, class_names=["Non-CKD", "CKD"],
                     filled=True, rounded=True, impurity=False, fontsize=fontsize, ax=ax)
    for ann in anns:
        text = ann.get_text()
        m = re.search(r"value = \[([\d.]+), ([\d.]+)\]", text)
        if not m:
            continue
        non, ckd = float(m.group(1)), float(m.group(2))
        lines = []
        split = re.match(r"(.+) <= ([\d.]+)", text.split("\n")[0])
        if split:
            feat, thr = split.group(1), float(split.group(2))
            lines.append(f"{names.get(feat, feat)} {readable_split(feat, thr, raw)}?")
        lines.append(f"Non-CKD {int(non)} | CKD {int(ckd)}")
        if not split:
            lines.append("→ CKD" if ckd > non else "→ Non-CKD")
        ann.set_text("\n".join(lines))
        frac = ckd / (non + ckd)
        color = COLORS["CKD"] if frac >= 0.5 else COLORS["Non-CKD"]
        ann.get_bbox_patch().set_facecolor(_blend(color, 0.15 + 0.6 * abs(frac - 0.5) * 2))
    return anns


# Colours for the two final models (kept separate from the CKD / non-CKD colours)
MODEL_COLORS = {"Full model": "#6250d6", "Cheap-test model": "#1baf7a"}
MODEL_STYLES = {"Full model": "-", "Cheap-test model": "--"}
