"""Convert RoboTwin HDF5 demos into a multi-camera ZARR for A2A training.

Layout written:
    data/head_camera   (N, 3, H, W) uint8
    data/left_camera   (N, 3, H, W) uint8        (if present in HDF5)
    data/right_camera  (N, 3, H, W) uint8        (if present in HDF5)
    data/state         (N, D)       float32
    data/action        (N, D)       float32
    meta/episode_ends  (E,)         int64
    meta attrs: task_name, task_config, expert_data_num, cam_keys, action_dim

Action convention follows policy/DP/process_data.py:
    state[t]  = joint_action/vector[t]
    action[t] = joint_action/vector[t+1]

Usage:
    python process_data.py <task_name> <task_config> <expert_data_num>
"""

import argparse
import os
import shutil

import cv2
import h5py
import numpy as np
import zarr

CAM_PRIORITY = ["head_camera", "left_camera", "right_camera"]


def load_episode(path: str):
    with h5py.File(path, "r") as root:
        vector = root["/joint_action/vector"][()]
        image_dict = {}
        if "/observation" in root:
            for cam in CAM_PRIORITY:
                key = f"/observation/{cam}/rgb"
                if key in root:
                    image_dict[cam] = root[key][()]
    return vector, image_dict


def decode_rgb(buffer: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode returned None; data may be corrupt")
    return img  # H,W,3 (BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument(
        "--data_root",
        type=str,
        default="../../data",
        help="RoboTwin data root, relative to policy/A2A/",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="./data",
        help="ZARR output root, relative to policy/A2A/",
    )
    args = parser.parse_args()

    load_dir = os.path.join(args.data_root, args.task_name, args.task_config)
    save_dir = os.path.join(
        args.save_root, f"{args.task_name}-{args.task_config}-{args.expert_data_num}.zarr"
    )
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    cam_buffers = {cam: [] for cam in CAM_PRIORITY}
    state_buffer = []
    action_buffer = []
    episode_ends = []
    total = 0
    cam_keys = None

    for ep in range(args.expert_data_num):
        path = os.path.join(load_dir, "data", f"episode{ep}.hdf5")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing HDF5 {path}. Run collect_data.sh first."
            )
        print(f"processing episode {ep + 1} / {args.expert_data_num}", end="\r")
        vector, image_dict = load_episode(path)
        if cam_keys is None:
            cam_keys = [c for c in CAM_PRIORITY if c in image_dict]
            if not cam_keys:
                raise RuntimeError(f"No supported cameras in {path}")
        else:
            missing = [c for c in cam_keys if c not in image_dict]
            if missing:
                raise RuntimeError(f"Episode {ep} missing cameras {missing}")

        n = vector.shape[0]
        if n < 2:
            continue
        # state[j] = vector[j], action[j] = vector[j+1]; drop the last frame
        for j in range(n - 1):
            for cam in cam_keys:
                cam_buffers[cam].append(decode_rgb(image_dict[cam][j]))
            state_buffer.append(vector[j])
            action_buffer.append(vector[j + 1])
        total += n - 1
        episode_ends.append(total)

    print()
    if total == 0:
        raise RuntimeError("No frames collected; aborting zarr write.")

    state_arr = np.asarray(state_buffer, dtype=np.float32)
    action_arr = np.asarray(action_buffer, dtype=np.float32)
    episode_ends_arr = np.asarray(episode_ends, dtype=np.int64)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    for cam in cam_keys:
        cam_arr = np.asarray(cam_buffers[cam])
        cam_arr = np.moveaxis(cam_arr, -1, 1)  # NHWC -> NCHW
        zarr_data.create_dataset(
            cam,
            data=cam_arr,
            chunks=(100, *cam_arr.shape[1:]),
            dtype=cam_arr.dtype,
            overwrite=True,
            compressor=compressor,
        )
        del cam_arr  # free memory

    zarr_data.create_dataset(
        "state",
        data=state_arr,
        chunks=(100, state_arr.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=action_arr,
        chunks=(100, action_arr.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arr,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )

    zarr_meta.attrs["task_name"] = args.task_name
    zarr_meta.attrs["task_config"] = args.task_config
    zarr_meta.attrs["expert_data_num"] = args.expert_data_num
    zarr_meta.attrs["cam_keys"] = cam_keys
    zarr_meta.attrs["action_dim"] = int(state_arr.shape[1])

    print(
        f"Wrote {save_dir}: {total} frames, {len(episode_ends)} episodes, "
        f"action_dim={state_arr.shape[1]}, cams={cam_keys}"
    )


if __name__ == "__main__":
    main()
