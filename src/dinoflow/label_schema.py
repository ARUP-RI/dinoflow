# label_schema.py
IGNORE_INDEX = -100

TASKS = {
    # binary
    "action_required": {"type": "binary"},
    "abnormal_population": {"type": "binary"},
    #"flow_indeterminate": {"type": "binary"},
    "suboptimal_viability": {"type": "binary"},
    "myeloid_lineage": {"type": "binary"},
    "bcell_lineage": {"type": "binary"},
    "t_nk_lineage": {"type": "binary"},

    # multiclass
    "malignant_vs_reactive": {
        "type": "multiclass",
        "classes": ["malignant", "reactive", "none"],
    },
    "clonality": {
        "type": "multiclass",
        "classes": ["clonal", "polyclonal", "uncertain", "none"],
    },
    "maturation": {
        "type": "multiclass",
        "classes": ["acute", "mature", "none"],
    },

    # ordinal
    "aberrancy_grade": {"type": "ordinal", "classes": [0, 1, 2]},  # ordinal over 0/1/2
}

# This is where you say which dataframe column provides each task's label.
# By default you can make it identical to the task name, but this allows renames.
LABEL_COL = {
    "abnormal_population": "abnormal_population",
    "suboptimal_viability": "suboptimal_viability",
    "malignant_vs_reactive": "malignant_vs_reactive",
    "myeloid_lineage": "lineage_myeloid",
    "b_cell_lineage": "lineage_B",
    "t_nk_lineage": "lineage_T",
    "clonality": "clonality",
    "acute_maturation": "maturation",
    "aberrancy": "aberrancy_grade",
    "action_required": "ACTION_REQUIRED",
}
