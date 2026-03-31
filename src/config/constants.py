TRAJ_INDEX = {
    "X": 0,
    "Y": 1,
    "PEN_CLASS": 2,
}


PEN_CLASS = {
    "PM": 0,
    "PU": 1,
    "CURSIVE_EOC": 2,
    "EOC": 3,
}


TRAJ_INDEX_EXPANDED = {
    "X": 0,
    "Y": 1,
    "PM": 2,
    "PU": 3,
    "CURSIVE_EOC": 4,
    "EOC": 5,
}

TRAJ_DIM = 3
TRAJ_DIM_EXPANDED = 6
PEN_STATE_DIM = 4


PEN_STATE_RANGE = slice(2, 6)


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SPACE_TOKEN = " "

PAD_ID   = 0
UNK_ID   = 1
SPACE_ID = 2

__all__ = [
    "TRAJ_INDEX", "TRAJ_INDEX_EXPANDED", "TRAJ_DIM", "TRAJ_DIM_EXPANDED",
    "PEN_STATE_DIM", "PEN_STATE_RANGE", "PEN_CLASS",
    "PAD_TOKEN", "UNK_TOKEN", "SPACE_TOKEN",
    "PAD_ID", "UNK_ID", "SPACE_ID",
]
