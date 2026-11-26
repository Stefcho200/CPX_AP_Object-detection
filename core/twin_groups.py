"""Defines groups of visually-near-identical ('twin') CPX products that
cannot be reliably told apart by the segmentation model alone and
require OCR-based label reading to disambiguate.

NOTE: not every listed product is necessarily trained/present in the
current dataset yet. group_for_label() only returns a group if the
locked label itself is a recognized member, so untrained placeholders
are harmless and future-proof this list for when more products are
added later.

Products not listed in any group below are assumed visually unique
and never trigger OCR disambiguation (e.g. CPX-AP-I-PN-M12,
CPX-AP-I-4DI-M8-3P, CPX-AP-I-PB-M12).
"""

# Twin groups now live in project.json ("twin_groups") so a different
# object set can be used without touching code. Set it to [] for objects
# that are visually distinct.
from core.project_config import twin_groups as _load_twin_groups

TWIN_GROUPS = _load_twin_groups()

# Products confirmed visually unique (no OCR disambiguation needed).
# Kept here purely for documentation/reference - group_for_label()
# already returns None for these automatically since they appear in
# no group above.
UNIQUE_PRODUCTS = [
    "CPX-AP-I-PN-M12",
    "CPX-AP-I-4DI-M8-3P",
    "CPX-AP-I-PB-M12",
]




def is_twin_label(label):
    return group_for_label(label) is not None


def all_grouped_labels():
    """Flat set of every label that belongs to some twin group."""
    result = set()
    for group in TWIN_GROUPS:
        result |= group
    return result


# core/twin_groups.py  -- add this

def build_class_remap(all_products):
    """Maps every product name to its 'training class' name - twin
    group members all collapse to one shared meta-class name, unique
    products map to themselves. Used only when constructing the
    dataset.yaml / labels for training - has zero effect on inference-
    time OCR disambiguation, which still uses the real product names."""
    remap = {}
    for group in TWIN_GROUPS:
        meta_name = "TWIN_" + "_OR_".join(sorted(group))
        for member in group:
            remap[member] = meta_name
    for product in all_products:
        if product not in remap:
            remap[product] = product  # unique products keep their own class
    return remap


# --- add to the bottom of core/twin_groups.py ---

# Fixed, distinct colors per twin group so they're always visually
# recognizable across sessions (not re-randomized on each app launch).
GROUP_COLORS = [
    "#3498db",  # blue      - M8-3P group
    "#9b59b6",  # purple    - M12-5P group
    "#1abc9c",  # teal      - 16NDI/16DI M8 group
    "#e67e22",  # burnt orange - 16DI/16DIO M12 group
    "#2ecc71",  # green     - EC/EP/CCB group
    "#e74c3c",  # red       - 4AI/4AI4AO group
]

UNIQUE_COLOR = "#7f8c8d"  # neutral gray for visually-unique products


def group_color(group_index):
    return GROUP_COLORS[group_index % len(GROUP_COLORS)]


def shade_color(hex_color, factor):
    """Lightens (factor > 1) or darkens (factor < 1) a hex color, used
    to give each member of an expanded twin group a distinguishable
    shade of the group's base color."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return f"#{r:02x}{g:02x}{b:02x}"


def meta_class_name(group):
    """Given a twin group (set), returns the exact meta-class name
    build_class_remap() would have assigned it - must stay in sync
    with build_class_remap()'s naming logic."""
    return "TWIN_" + "_OR_".join(sorted(group))


def group_for_label(label):
    """Returns the twin group (as a set) that `label` belongs to, or
    None if `label` is not part of any twin group. Recognizes BOTH
    real product names (pre-merge / legacy models) and merged
    meta-class names (post-merge models), so this keeps working
    regardless of which model version is currently loaded."""
    for group in TWIN_GROUPS:
        if label in group:
            return group
        if label == meta_class_name(group):
            return group
    return None


def build_display_entries(counts):
    """Builds chart-ready entries from a {product: count} dict.
    Each entry is a dict:
        {
            "key": unique id for this row,
            "label": display label,
            "count": total count for this row,
            "color": base color,
            "expandable": bool,
            "members": [(product_name, count, shade_color), ...]  # only if expandable
        }
    Twin group members are summed into one entry per group, colored
    consistently; unique products get their own entry in neutral gray.
    Products with zero recorded images anywhere are simply omitted
    (nothing to show)."""
    entries = []
    grouped_products = set()

    for i, group in enumerate(TWIN_GROUPS):
        base_color = group_color(i)
        members = []
        total = 0
        for product in sorted(group):
            if product in counts:
                members.append((product, counts[product]))
                total += counts[product]
                grouped_products.add(product)

        if not members:
            continue  # none of this group's products have been recorded yet

        # Shade each member a bit differently so they're tellable apart
        # when expanded, while staying clearly the same color family.
        shaded_members = []
        n = len(members)
        for idx, (product, count) in enumerate(members):
            # Spread shades from slightly darker to slightly lighter.
            factor = 0.7 + (idx / max(1, n - 1)) * 0.8 if n > 1 else 1.0
            shaded_members.append((product, count, shade_color(base_color, factor)))

        short_names = [m[0].replace("CPX-AP-I-", "") for m in members]
        entries.append({
            "key": f"group_{i}",
            "label": "TWIN: " + " / ".join(short_names),
            "count": total,
            "color": base_color,
            "expandable": True,
            "members": shaded_members,
        })

    for product, count in counts.items():
        if product in grouped_products:
            continue
        entries.append({
            "key": f"unique_{product}",
            "label": product,
            "count": count,
            "color": UNIQUE_COLOR,
            "expandable": False,
            "members": [],
        })

    return entries
