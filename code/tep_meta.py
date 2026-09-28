# -*- coding: utf-8 -*-
"""TEP metadata used by the diagnostic evaluation."""
TARGET_VAR = 35
XMEAS_LIST = [i for i in range(1, 42) if i != TARGET_VAR]
NODE_TO_XMEAS = {i: x for i, x in enumerate(XMEAS_LIST)}
XMEAS_TO_NODE = {x: i for i, x in NODE_TO_XMEAS.items()}
NUM_NODES = 40
ACTION_DIM = 11

UNIT_OF_XMEAS = {}
def _assign(unit, ids):
    for i in ids:
        UNIT_OF_XMEAS[i] = unit

_assign("FEED",       [1,2,3,4])
_assign("REACTOR",    [6,7,8,9] + list(range(23,29)))
_assign("RCOOL",      [21])
_assign("SEPARATOR",  [11,12,13,14])
_assign("SCOOL",      [22])
_assign("COMPRESSOR", [5,20])
_assign("PURGE",      [10] + list(range(29,37)))
_assign("STRIPPER",   [15,16,17,18,19])
_assign("PRODUCT",    [37,38,39,40,41])

UNIT_OF_XMV = {
    1:"FEED", 2:"FEED", 3:"FEED", 4:"FEED",
    5:"COMPRESSOR", 6:"PURGE", 7:"SEPARATOR",
    8:"STRIPPER", 9:"STRIPPER", 10:"RCOOL", 11:"SCOOL",
}
UNITS = [
    "FEED","REACTOR","RCOOL","SEPARATOR","SCOOL",
    "COMPRESSOR","PURGE","STRIPPER","PRODUCT",
]
ROOT_UNITS = ["FEED","REACTOR","RCOOL","SCOOL"]
VIRTUAL_TARGET_XMEAS = 35
VIRTUAL_TARGET_UNIT = "PURGE"
VIRTUAL_TARGET_LABEL = "Y35"

# Unit-level ground truth for the 13 evaluated scenarios.
FAULT_GT = {
    1:  dict(name="A/C feed ratio, B composition constant", gt_unit="FEED"),
    2:  dict(name="B composition, A/C ratio constant", gt_unit="FEED"),
    3:  dict(name="D feed temperature", gt_unit="FEED"),
    4:  dict(name="Reactor cooling water inlet temperature", gt_unit="RCOOL"),
    5:  dict(name="Condenser cooling water inlet temperature", gt_unit="SCOOL"),
    6:  dict(name="A feed loss", gt_unit="FEED"),
    7:  dict(name="C header pressure loss", gt_unit="FEED"),
    8:  dict(name="A/B/C feed composition", gt_unit="FEED"),
    10: dict(name="C feed temperature", gt_unit="FEED"),
    11: dict(name="Reactor cooling water inlet temperature (random)", gt_unit="RCOOL"),
    13: dict(name="Reaction kinetics", gt_unit="REACTOR"),
    14: dict(name="Reactor cooling water valve sticking", gt_unit="RCOOL"),
    15: dict(name="Condenser cooling water valve sticking", gt_unit="SCOOL"),
}

# Reference routes for the XMEAS35/PURGE evaluation.
PRIMARY_REF = {
    "FEED":    ["FEED","REACTOR","SEPARATOR","PURGE"],
    "RCOOL":   ["RCOOL","REACTOR","SEPARATOR","PURGE"],
    "SCOOL":   ["SCOOL","SEPARATOR","PURGE"],
    "REACTOR": ["REACTOR","SEPARATOR","PURGE"],
}

ALLOWED_UNIT_EDGES = {
    ("FEED","REACTOR"),
    ("RCOOL","REACTOR"),
    ("COMPRESSOR","REACTOR"),
    ("STRIPPER","REACTOR"),
    ("REACTOR","SEPARATOR"),
    ("SCOOL","SEPARATOR"),
    ("SEPARATOR","COMPRESSOR"),
    ("SEPARATOR","PURGE"),
    ("SEPARATOR","STRIPPER"),
    ("STRIPPER","PRODUCT"),
}

# Unit-level process-influence reference graph (13 edges).
# Used for the directed edge precision / recall / F1 reported in Section 5.3.
# PRIMARY_REF above is retained unchanged: it supplies the reference routes used
# by the root-consistent path metrics of Section 5.4.3 and by ordered path
# recovery. The shortest routes of REFERENCE_GRAPH coincide with PRIMARY_REF.
REFERENCE_GRAPH = {
    # direct material connections (6)
    ("FEED", "REACTOR"),
    ("REACTOR", "SEPARATOR"),
    ("SEPARATOR", "COMPRESSOR"),
    ("COMPRESSOR", "REACTOR"),
    ("SEPARATOR", "PURGE"),
    ("SEPARATOR", "STRIPPER"),
    # energy coupling (2)
    ("SCOOL", "SEPARATOR"),
    ("RCOOL", "REACTOR"),
    # recycle feedback (2)
    ("SEPARATOR", "REACTOR"),
    ("COMPRESSOR", "PURGE"),
    # indirect process propagation (3)
    ("REACTOR", "PURGE"),
    ("FEED", "COMPRESSOR"),
    ("RCOOL", "STRIPPER"),
}

REFERENCE_GRAPH_CATEGORIES = {
    "direct_material": [
        ("FEED", "REACTOR"), ("REACTOR", "SEPARATOR"), ("SEPARATOR", "COMPRESSOR"),
        ("COMPRESSOR", "REACTOR"), ("SEPARATOR", "PURGE"), ("SEPARATOR", "STRIPPER"),
    ],
    "energy_coupling": [("SCOOL", "SEPARATOR"), ("RCOOL", "REACTOR")],
    "recycle_feedback": [("SEPARATOR", "REACTOR"), ("COMPRESSOR", "PURGE")],
    "indirect_propagation": [("REACTOR", "PURGE"), ("FEED", "COMPRESSOR"), ("RCOOL", "STRIPPER")],
}

def unit_of_node(n):
    return UNIT_OF_XMEAS[NODE_TO_XMEAS[int(n)]]
