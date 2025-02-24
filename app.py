import streamlit as st
st.set_page_config(page_title="SPECT Visualization", layout="wide")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
import yaml
import h5py
import io
from typing import TypedDict


class Transform(TypedDict):
    angle: float
    trans_r: float
    trans_t: float

#All the Utility functions
def transform_verts(verts: np.ndarray, trans: Transform) -> np.ndarray:
    angle = np.deg2rad(trans["angle"])
    trans_r, trans_t = trans["trans_r"], trans["trans_t"]
    mtrans = np.array([[np.cos(angle), -np.sin(angle)], 
                      [np.sin(angle), np.cos(angle)]])
    return np.array([mtrans @ (vert + np.array([trans_r, trans_t])) for vert in verts])

def geom2verts(geom: np.ndarray, trans: Transform) -> np.ndarray:
    verts = np.array([
        [geom[0], geom[2]],
        [geom[1], geom[2]],
        [geom[1], geom[3]],
        [geom[0], geom[3]],
        [geom[0], geom[2]],
    ])
    return transform_verts(verts, trans)

def verts_to_patch(verts: np.ndarray) -> patches.PathPatch:
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    return patches.PathPatch(Path(verts, codes), facecolor="orange", ec="none")

def geoms_to_patchcollection(geoms: np.ndarray, trans_list: list[Transform], 
                          fc: str = "orange", ec: str = "none") -> PatchCollection:
    verts_list = [geom2verts(geom, trans) for trans in trans_list for geom in geoms]
    return PatchCollection([verts_to_patch(v) for v in verts_list], fc=fc, ec=ec)

def get_det_geoms(config: dict) -> np.ndarray:
    indices = np.array(config["detector"]["active geometry indices"], dtype=np.int32)
    geoms = np.array(config["detector"]["detector geometry"])
    return np.array([geoms[geoms[:, 6] == idx][0] for idx in indices])

def get_geom_center_xy(geom: np.ndarray) -> np.ndarray:
    return (geom[:4:2] + geom[1:4:2]) * 0.5

# Streamlit UI
st.title("Detector Probability Density Visualization")

# File upload in sidebar
with st.sidebar:
    st.header("File Upload")
    yaml_file = st.file_uploader("Upload YAML Config", type=["yaml"])
    hdf5_file = st.file_uploader("Upload HDF5 System Matrix", type=["hdf5"])

if yaml_file and hdf5_file:
    try:
        config = yaml.safe_load(yaml_file)
        geoms = np.array(config["detector"]["detector geometry"])
        
        hdf5_content = hdf5_file.read()
        
        with h5py.File(io.BytesIO(hdf5_content), "r") as f:
            data = f["system matrix"]
            max_detectors = 144
            detector_id = st.slider("Select Detector ID", 0, max_detectors-1, 0)
            col1, col2 = st.columns(2)
            with col1:
                show_apertures = st.checkbox("Show Aperture Markers", True)
            with col2:
                line_alpha = st.slider("Connection Line Alpha", 0.0, 1.0, 0.7)

            if st.button("Generate Visualization"):
                with st.spinner("Creating visualization..."):
                    fov_dims = np.array(config["FOV"]["N voxels xyz"]) * np.array(config["FOV"]["mm per voxel xyz"])
                    trans_t = -(np.max(geoms[:, 3]) + np.min(geoms[:, 2])) / 2
                    det_dims = np.array([
                        np.max(geoms[:, 1]) - np.min(geoms[:, 0]),
                        np.max(geoms[:, 3]) - np.min(geoms[:, 2]),
                        np.max(geoms[:, 5]) - np.min(geoms[:, 4]),
                    ])
                    
                    trans_r = config["relation"]["radial shift"]["data"][0]
                    trans_list = [{
                        "angle": config["relation"]["rotation"]["data"][0],
                        "trans_r": trans_r,
                        "trans_t": trans_t
                    }]
                    fig, ax = plt.subplots(figsize=(15, 10), dpi=150)
                    fov_nvx = np.array(config["FOV"]["N voxels xyz"])[:2]
                    ppdf = data[detector_id].reshape((fov_nvx[1], fov_nvx[0]))
                    ax.add_patch(patches.Rectangle(
                        -fov_dims[:2] * 0.5, fov_dims[0], fov_dims[1],
                        fc='none', ec='k', ls='--'
                    ))
                    active_det_geoms = geoms[geoms[:, 6] != 0]
                    plate_geoms = geoms[geoms[:, 6] == 0]
                    det_coll = geoms_to_patchcollection(active_det_geoms, trans_list)
                    plate_coll = geoms_to_patchcollection(plate_geoms, trans_list, fc="gray")
                    ax.add_collection(det_coll)
                    ax.add_collection(plate_coll)
                    ax.set_xlim((fov_dims[0]) * (-1.1), (trans_r + det_dims[0]) * 1.1)
                    ax.set_ylim((det_dims[0]) * (-1.1), (det_dims[0]) * 1.1)
                    ax.set_aspect("equal")
                    det_geoms = get_det_geoms(config)
                    current_geom = np.array([det_geoms[detector_id]])
                    current_coll = geoms_to_patchcollection(current_geom, trans_list, fc="red")
                    ax.add_collection(current_coll)
                    geom_center = get_geom_center_xy(det_geoms[detector_id])
                    geom_center_xy = transform_verts([geom_center], trans_list[0])
                    
                    aperture_y = (plate_geoms[:-1, 3] + plate_geoms[1:, 2]) * 0.5 + trans_t
                    aperture_centers = np.stack(
                        (np.full(aperture_y.shape[0], trans_r + 0.5), aperture_y)
                    ).T

                    if show_apertures:
                        ax.plot(aperture_centers[:, 0], aperture_centers[:, 1], 
                                "o", ms=1, color='blue', zorder=4)
                        
                        clipbox = patches.Rectangle(
                            (ax.get_xlim()[0], ax.get_ylim()[0]),
                            geom_center_xy[0][0] - ax.get_xlim()[0],
                            ax.get_ylim()[1] - ax.get_ylim()[0],
                            transform=ax.transData,
                            visible=False
                        )
                        
                        for pA in aperture_centers:
                            line = ax.axline(geom_center_xy[0], pA, ls="--", lw=0.5,
                                           color='gray', alpha=line_alpha, zorder=3)
                            line.set_clip_path(clipbox)

                    im = ax.imshow(ppdf.T, extent=(-fov_dims[0]*0.5, fov_dims[0]*0.5, 
                                                 -fov_dims[1]*0.5, fov_dims[1]*0.5),
                                 origin="lower")
                    cbar = fig.colorbar(im, ax=ax, location="left", pad=0.07)
                    cbar.set_label("Probability Density", rotation=90)
                    ax.set_xlabel("Transverse plane x (mm)", fontsize=14)
                    ax.set_ylabel("Transverse plane y (mm)", fontsize=14)
                    ax.set_title(f"Detector {detector_id:03d}")
                    
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
else:
    st.info("Please upload both YAML configuration and HDF5 system matrix files to begin.")