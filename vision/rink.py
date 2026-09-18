"""Define the provisional center-origin NHL rink landmarks used by labeling."""

# Naming convention: feature_END_SIDE, where E/W select rink end and N/S side.
LANDMARKS = {
    "center": (0.0, 0.0),
    "centerBrd_N": (0.0, 42.5), "centerBrd_S": (0.0, -42.5),
    "blue_E_N": (25.0, 42.5), "blue_E_S": (25.0, -42.5),
    "blue_W_N": (-25.0, 42.5), "blue_W_S": (-25.0, -42.5),
    "dotNZ_E_N": (20.0, 22.0), "dotNZ_E_S": (20.0, -22.0),
    "dotNZ_W_N": (-20.0, 22.0), "dotNZ_W_S": (-20.0, -22.0),
    "dotEZ_E_N": (69.0, 22.0), "dotEZ_E_S": (69.0, -22.0),
    "dotEZ_W_N": (-69.0, 22.0), "dotEZ_W_S": (-69.0, -22.0),
    "post_E_N": (89.0, 3.0), "post_E_S": (89.0, -3.0),
    "post_W_N": (-89.0, 3.0), "post_W_S": (-89.0, -3.0),
    "trapGL_E_N": (89.0, 11.0), "trapGL_E_S": (89.0, -11.0),
    "trapGL_W_N": (-89.0, 11.0), "trapGL_W_S": (-89.0, -11.0),
    "trapEB_E_N": (100.0, 14.0), "trapEB_E_S": (100.0, -14.0),
    "trapEB_W_N": (-100.0, 14.0), "trapEB_W_S": (-100.0, -14.0),
    "goalBrd_E_N": (89.0, 36.75), "goalBrd_E_S": (89.0, -36.75),
    "goalBrd_W_N": (-89.0, 36.75), "goalBrd_W_S": (-89.0, -36.75),
}
