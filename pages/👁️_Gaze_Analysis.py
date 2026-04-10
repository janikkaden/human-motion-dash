import glob
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make gaze-analysis modules importable
_GAZE_DIR = os.path.join(os.path.dirname(__file__), "..", "gaze-analysis")
if _GAZE_DIR not in sys.path:
    sys.path.insert(0, _GAZE_DIR)

from obstacle_classes import ObstacleCollection  # noqa: E402

st.set_page_config(
    page_title="Gaze Analysis",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GAZE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "gaze-analysis"))


SCENES = {
    "Scenario 4 - Layout 2": {
        "day": 31,
        "stem": "250131_1529_nav_4p_r2",
        "scene_short": "nav_4p_r2",
        "participants": ["P1", "P3", "P4"],
    },
    "Scenario 5 - Special exhibit": {
        "day": 30,
        "stem": "2025-01-30_1630_obstacles_3p_sp_1",
        "scene_short": "obstacles_3p_sp_1",
        "participants": ["P1", "P2", "P3"],
    },
}

GAZE_COLOR_MAP = {
    "precomputed_fallback": "blue",
    "yolo_matched": "yellow",
    "yolo_no_match": "gray",
    "yolo_refined": "green",
}

GAZE_MODE_LABELS = {
    "precomputed_fallback": "Interpolated",
    "yolo_matched": "Refined using YOLO",
    "yolo_no_match": "YOLO No Match",
    "yolo_refined": "Refined using YOLO2/SAM2",
}

# ---------------------------------------------------------------------------
# Data / media loaders
# ---------------------------------------------------------------------------


@st.cache_data
def load_intersection_data(stem: str, participant: str) -> pd.DataFrame:
    path = os.path.join(GAZE_DIR, "Intersection_Data_3D", f"{stem}_{participant}.csv")
    return pd.read_csv(path)


@st.cache_resource
def load_scene_obstacles(day: int) -> ObstacleCollection:
    path = os.path.join(GAZE_DIR, f"obstacles{day}.json")
    return ObstacleCollection.load_json(path)


def find_media_files(scene_short: str, participant: str) -> dict:
    """Glob for MP4 files for the given scene/participant."""
    files = {}
    hits = glob.glob(os.path.join(GAZE_DIR, "videos", f"tobii_*{participant}*{scene_short}*_data.mp4"))
    if hits:
        files["mp4"] = hits[0]
    return files


# ---------------------------------------------------------------------------
# 3D figure helpers
# ---------------------------------------------------------------------------


def build_obstacle_traces(collection: ObstacleCollection) -> list:
    traces = []
    for obs in collection.obstacles:
        pts = obs.base_points
        xs = list(pts[:, 0]) + [pts[0, 0]]
        ys = list(pts[:, 1]) + [pts[0, 1]]

        # Semi-transparent solid fill — triangulate using the obstacle's own geometry
        verts = obs.get_vertices_3d()   # (n_verts, 3)
        faces = obs.get_faces()         # list of face index lists (quads / polygons)
        ii, jj, kk = [], [], []
        for face in faces:
            for idx in range(1, len(face) - 1):
                ii.append(face[0])
                jj.append(face[idx])
                kk.append(face[idx + 1])
        traces.append(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=ii, j=jj, k=kk,
            color="steelblue",
            opacity=0.25,
            name=obs.obstacle_type,
            showlegend=False,
            flatshading=True,
            hovertemplate=f"<b>{obs.obstacle_type}</b><extra></extra>",
        ))

        # Wireframe edges on top
        for z_level in [obs.z_min, obs.z_max]:
            traces.append(go.Scatter3d(
                x=xs, y=ys, z=[z_level] * len(xs),
                mode="lines",
                line=dict(color="steelblue", width=2),
                name=obs.obstacle_type,
                showlegend=False,
                hovertemplate=f"<b>{obs.obstacle_type}</b><br>z={z_level:.0f} mm<extra></extra>",
            ))

        for x, y in zip(xs[:-1], ys[:-1]):
            traces.append(go.Scatter3d(
                x=[x, x], y=[y, y], z=[obs.z_min, obs.z_max],
                mode="lines",
                line=dict(color="steelblue", width=1),
                showlegend=False,
                hoverinfo="skip",
            ))
    return traces


def build_full_trajectory_fig(
    df: pd.DataFrame,
    obstacles: ObstacleCollection | None,
    gaze_modes: list,
    show_head: bool,
) -> go.Figure:
    fig = go.Figure()

    if obstacles:
        for t in build_obstacle_traces(obstacles):
            fig.add_trace(t)

    if show_head:
        head = df.dropna(subset=["head_position_x", "head_position_y", "head_position_z"])
        if not head.empty:
            fig.add_trace(go.Scatter3d(
                x=head["head_position_x"],
                y=head["head_position_y"],
                z=head["head_position_z"],
                mode="lines",
                line=dict(color="orange", width=4),
                name="Head trajectory",
                opacity=0.5,
            ))

    gaze = df[df["intersection_type"] != "none"].dropna(subset=["intersection_x"])
    if gaze_modes:
        gaze = gaze[gaze["intersection_type"].isin(gaze_modes)]

    for mode, group in gaze.groupby("intersection_type"):
        fig.add_trace(go.Scatter3d(
            x=group["intersection_x"],
            y=group["intersection_y"],
            z=group["intersection_z"],
            mode="markers",
            marker=dict(size=3, color=GAZE_COLOR_MAP.get(mode, "white"), opacity=0.6),
            name=GAZE_MODE_LABELS.get(mode, mode),
            text=group["object_type"].fillna("unknown"),
            hovertemplate="<b>%{text}</b><br>x=%{x:.0f} y=%{y:.0f} z=%{z:.0f}<extra></extra>",
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
            aspectmode="data",
            camera=dict(eye=dict(x=0, y=-2.5, z=1.5)),
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, itemsizing="constant"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=650,
    )
    return fig


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("👁️ Gaze Analysis")

with st.expander("About this page"):
    st.markdown(
        """
        Visualises precomputed gaze-obstacle intersection data from the
        **Tobii Pro Glasses 3 + Vicon + YOLO/SAM2** pipeline.

        | Color | Mode | Meaning |
        |-------|------|---------|
        | 🔵 Blue | Interpolated | 3D intersection from Vicon data only (Schreiter et al.) |
        | 🟡 Yellow | Refined using YOLO | Gaze aligns with a detected object |
        | ⚫ Gray | YOLO No Match | Detection found but doesn't match 3D result |
        | 🟢 Green | Refined using YOLO2/SAM2 | Refined with SAM2 segmentation mask |

        All spatial values are in **millimetres** (Z-up, Vicon world frame).
        """
    )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Scene selection")
    scene_label = st.selectbox("Scene", list(SCENES.keys()))
    scene_cfg = SCENES[scene_label]
    participant_label = st.selectbox("Participant", scene_cfg["participants"])
    participant = participant_label.lower()  # p1/p2/p3/p4

    st.markdown("---")
    st.subheader("Gaze modes")
    gaze_modes = st.multiselect(
        "Show modes",
        options=list(GAZE_COLOR_MAP.keys()),
        default=["precomputed_fallback", "yolo_matched", "yolo_refined"],
        format_func=lambda m: GAZE_MODE_LABELS[m],
    )

    st.markdown("---")
    show_head = st.checkbox("Show head trajectory", value=True)
    show_obstacles = st.checkbox("Show room obstacles", value=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
with st.spinner("Loading data…"):
    try:
        df = load_intersection_data(scene_cfg["stem"], participant)
    except FileNotFoundError:
        st.error(
            f"Intersection data not found for **{participant_label}** in **{scene_label}**. "
            "Make sure CSV files exist in `gaze-analysis/Intersection_Data_3D/`."
        )
        st.stop()

    obstacles = None
    if show_obstacles:
        try:
            obstacles = load_scene_obstacles(scene_cfg["day"])
        except FileNotFoundError:
            st.warning(f"Obstacle layout for **{scene_label}** not found — obstacles skipped.")

media = find_media_files(scene_cfg["scene_short"], participant)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_3d, tab_anim, tab_stats = st.tabs(["📊 3D View", "🎬 3D Animation", "📈 Statistics"])

# ── Tab 1: full trajectory 3D view ─────────────────────────────────────────
with tab_3d:
    frame_col = df["frame"] if "frame" in df.columns else pd.Series(df.index, index=df.index)
    f_min, f_max = int(frame_col.min()), int(frame_col.max())

    frame_range = st.slider(
        "Time range (frames)",
        min_value=f_min,
        max_value=f_max,
        value=(f_min, f_max),
        step=1,
    )

    df_filtered: pd.DataFrame = df.loc[frame_col.between(frame_range[0], frame_range[1])]  # type: ignore[assignment]

    with st.spinner("Building 3D figure…"):
        fig_full = build_full_trajectory_fig(df_filtered, obstacles, gaze_modes, show_head)
    st.plotly_chart(fig_full, use_container_width=True)

# ── Tab 2: animation player ────────────────────────────────────────────────
with tab_anim:
    if "mp4" not in media:
        st.info(
            f"No 3D animation video found for **{participant_label}** / **{scene_label}**. "
            "Run `visulaize3D_refactored.ipynb` to generate it."
        )
    else:
        st.caption("⚠️ Video is capped to the first 1000 frames.")
        st.markdown(
            "<style>[data-testid='stVideo'] { max-width: 1200px; }</style>",
            unsafe_allow_html=True,
        )
        st.video(media["mp4"])

# ── Tab 3: Statistics ───────────────────────────────────────────────────────
with tab_stats:
    total = len(df)
    on_obj = df[df["intersection_type"] != "none"].dropna(subset=["intersection_x"])
    pct = 100 * len(on_obj) / total if total > 0 else 0.0

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total frames", f"{total:,}")
    col_b.metric("Frames with gaze on object", f"{len(on_obj):,}")
    col_c.metric("Coverage", f"{pct:.1f}%")

    if not on_obj.empty:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Gaze hits by object**")
            by_obj = on_obj.groupby("object_type").size().sort_values(ascending=False)
            st.bar_chart(by_obj)

        with col_right:
            st.markdown("**Gaze hits by mode**")
            by_mode = on_obj.groupby("intersection_type").size().sort_values(ascending=False)
            by_mode.index = [GAZE_MODE_LABELS.get(m, m) for m in by_mode.index]
            st.bar_chart(by_mode)
