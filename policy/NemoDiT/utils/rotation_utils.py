import numpy as np
from typing import Union


def quaternion_to_rotation_matrix(quat: np.ndarray, convention: str = "wxyz") -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix.

    Args:
        quat: Quaternion array of shape (..., 4)
        convention: Quaternion convention, either "wxyz" or "xyzw"

    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    quat = np.asarray(quat)
    original_shape = quat.shape[:-1]
    quat = quat.reshape(-1, 4)

    if convention == "xyzw":
        # Convert xyzw to wxyz
        quat = quat[:, [3, 0, 1, 2]]
    elif convention != "wxyz":
        raise ValueError(f"Unknown quaternion convention: {convention}")

    # Normalize quaternion
    quat = quat / (np.linalg.norm(quat, axis=-1, keepdims=True) + 1e-8)

    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Compute rotation matrix elements
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)

    rotation_matrix = np.stack([
        np.stack([r00, r01, r02], axis=-1),
        np.stack([r10, r11, r12], axis=-1),
        np.stack([r20, r21, r22], axis=-1)
    ], axis=-2)

    return rotation_matrix.reshape(*original_shape, 3, 3)


def rotation_matrix_to_rot6d(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to 6D rotation representation (rot6d).

    Takes the first two columns of the rotation matrix.

    Args:
        rotation_matrix: Rotation matrix of shape (..., 3, 3)

    Returns:
        rot6d representation of shape (..., 6)
    """
    rotation_matrix = np.asarray(rotation_matrix)
    # Take first two columns and flatten
    rot6d = rotation_matrix[..., :, :2].reshape(*rotation_matrix.shape[:-2], 6)
    return rot6d


def quaternion_to_rot6d(quat: np.ndarray, convention: str = "wxyz") -> np.ndarray:
    """
    Convert quaternion directly to rot6d representation.

    Args:
        quat: Quaternion array of shape (..., 4)
        convention: Quaternion convention, either "wxyz" or "xyzw"

    Returns:
        rot6d representation of shape (..., 6)
    """
    rotation_matrix = quaternion_to_rotation_matrix(quat, convention)
    return rotation_matrix_to_rot6d(rotation_matrix)


def rot6d_to_rotation_matrix(rot6d: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation back to rotation matrix.

    Uses Gram-Schmidt orthogonalization to ensure valid rotation matrix.

    Args:
        rot6d: 6D rotation of shape (..., 6)

    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    rot6d = np.asarray(rot6d)
    original_shape = rot6d.shape[:-1]
    rot6d = rot6d.reshape(-1, 6)

    # Extract first two columns
    a1 = rot6d[:, :3]
    a2 = rot6d[:, 3:6]

    # Gram-Schmidt orthogonalization
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)

    rotation_matrix = np.stack([b1, b2, b3], axis=-1)
    return rotation_matrix.reshape(*original_shape, 3, 3)


def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray, convention: str = "wxyz") -> np.ndarray:
    """
    Convert rotation matrix to quaternion.

    Args:
        rotation_matrix: Rotation matrix of shape (..., 3, 3)
        convention: Output quaternion convention, "wxyz" or "xyzw"

    Returns:
        Quaternion of shape (..., 4)
    """
    rotation_matrix = np.asarray(rotation_matrix)
    original_shape = rotation_matrix.shape[:-2]
    rotation_matrix = rotation_matrix.reshape(-1, 3, 3)

    batch_size = rotation_matrix.shape[0]
    quat = np.zeros((batch_size, 4))

    for i in range(batch_size):
        R = rotation_matrix[i]
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        quat[i] = [w, x, y, z]

    # Normalize
    quat = quat / (np.linalg.norm(quat, axis=-1, keepdims=True) + 1e-8)

    if convention == "xyzw":
        quat = quat[:, [1, 2, 3, 0]]

    return quat.reshape(*original_shape, 4)


def rot6d_to_quaternion(rot6d: np.ndarray, convention: str = "wxyz") -> np.ndarray:
    """
    Convert 6D rotation representation to quaternion.

    Args:
        rot6d: 6D rotation of shape (..., 6)
        convention: Output quaternion convention, "wxyz" or "xyzw"

    Returns:
        Quaternion of shape (..., 4)
    """
    rotation_matrix = rot6d_to_rotation_matrix(rot6d)
    return rotation_matrix_to_quaternion(rotation_matrix, convention)


def convert_endpose_9d_to_7d(
    endpose_9d: np.ndarray,
    quat_convention: str = "wxyz"
) -> np.ndarray:
    """
    Convert 9D end effector pose (3 translation + 6 rot6d) back to
    7D representation (3 translation + 4 quaternion).

    Args:
        endpose_9d: End effector pose of shape (..., 9)
                    Format: [x, y, z, r1, r2, r3, r4, r5, r6]
        quat_convention: Output quaternion convention

    Returns:
        End effector pose of shape (..., 7)
    """
    endpose_9d = np.asarray(endpose_9d)
    translation = endpose_9d[..., :3]
    rot6d = endpose_9d[..., 3:9]

    quaternion = rot6d_to_quaternion(rot6d, quat_convention)
    return np.concatenate([translation, quaternion], axis=-1)


def convert_pose_quat_to_rot6d(
    translation: np.ndarray,
    quaternion: np.ndarray,
    quat_convention: str = "wxyz"
) -> np.ndarray:
    """
    Convert pose from (translation + quaternion) to (translation + rot6d).

    Args:
        translation: Translation array of shape (..., 3)
        quaternion: Quaternion array of shape (..., 4)
        quat_convention: Quaternion convention, either "wxyz" or "xyzw"

    Returns:
        Combined pose of shape (..., 9) as [x, y, z, r1, r2, r3, r4, r5, r6]
    """
    rot6d = quaternion_to_rot6d(quaternion, quat_convention)
    return np.concatenate([translation, rot6d], axis=-1)


def convert_endpose_7d_to_9d(
    endpose_7d: np.ndarray,
    quat_convention: str = "wxyz"
) -> np.ndarray:
    """
    Convert 7D end effector pose (3 translation + 4 quaternion) to
    9D representation (3 translation + 6 rot6d).

    Args:
        endpose_7d: End effector pose of shape (..., 7)
                    Format: [x, y, z, qw, qx, qy, qz] or [x, y, z, qx, qy, qz, qw]
        quat_convention: Quaternion convention in endpose_7d
                        "wxyz" means [x,y,z, w,x,y,z]
                        "xyzw" means [x,y,z, x,y,z,w]

    Returns:
        End effector pose of shape (..., 9)
        Format: [x, y, z, r1, r2, r3, r4, r5, r6]
    """
    endpose_7d = np.asarray(endpose_7d)
    translation = endpose_7d[..., :3]
    quaternion = endpose_7d[..., 3:7]

    return convert_pose_quat_to_rot6d(translation, quaternion, quat_convention)


class EndEffectorPose:
    """
    Helper class for end effector pose conversion.

    Example:
        pose = EndEffectorPose(
            translation=[x, y, z],
            rotation=[w, x, y, z],
            rotation_type="quat",
            quat_convention="wxyz"
        )
        model_input = pose.xyz_rot6d  # Returns [x, y, z, r1, r2, r3, r4, r5, r6]
    """

    def __init__(
        self,
        translation: Union[list, np.ndarray],
        rotation: Union[list, np.ndarray],
        rotation_type: str = "quat",
        quat_convention: str = "wxyz"
    ):
        """
        Initialize EndEffectorPose.

        Args:
            translation: 3D translation [x, y, z]
            rotation: Rotation representation (quaternion or rotation matrix)
            rotation_type: Type of rotation input ("quat" or "matrix")
            quat_convention: Quaternion convention if rotation_type is "quat"
        """
        self.translation = np.asarray(translation)
        self.rotation = np.asarray(rotation)
        self.rotation_type = rotation_type
        self.quat_convention = quat_convention

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get rotation as 3x3 matrix."""
        if self.rotation_type == "quat":
            return quaternion_to_rotation_matrix(self.rotation, self.quat_convention)
        elif self.rotation_type == "matrix":
            return self.rotation.reshape(3, 3)
        else:
            raise ValueError(f"Unknown rotation type: {self.rotation_type}")

    @property
    def rot6d(self) -> np.ndarray:
        """Get rotation as 6D representation."""
        return rotation_matrix_to_rot6d(self.rotation_matrix)

    @property
    def xyz_rot6d(self) -> np.ndarray:
        """Get full pose as 9D vector: [x, y, z, r1, r2, r3, r4, r5, r6]."""
        return np.concatenate([self.translation.flatten(), self.rot6d.flatten()])


# Testing
if __name__ == "__main__":
    print("Testing rotation utilities...")

    # Test quaternion to rot6d conversion
    # Identity quaternion [w, x, y, z] = [1, 0, 0, 0]
    quat_identity = np.array([1.0, 0.0, 0.0, 0.0])
    rot_matrix = quaternion_to_rotation_matrix(quat_identity, convention="wxyz")
    print(f"Identity quaternion -> rotation matrix:\n{rot_matrix}")

    rot6d = quaternion_to_rot6d(quat_identity, convention="wxyz")
    print(f"Identity quaternion -> rot6d: {rot6d}")

    # Test batch conversion
    batch_quat = np.random.randn(10, 4)
    batch_quat = batch_quat / np.linalg.norm(batch_quat, axis=-1, keepdims=True)
    batch_rot6d = quaternion_to_rot6d(batch_quat, convention="wxyz")
    print(f"Batch quaternion shape: {batch_quat.shape} -> rot6d shape: {batch_rot6d.shape}")

    # Test 7D to 9D conversion
    endpose_7d = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])  # [x,y,z, w,x,y,z]
    endpose_9d = convert_endpose_7d_to_9d(endpose_7d, quat_convention="wxyz")
    print(f"7D endpose: {endpose_7d}")
    print(f"9D endpose: {endpose_9d}")

    # Test EndEffectorPose class
    pose = EndEffectorPose(
        translation=[1.0, 2.0, 3.0],
        rotation=[1.0, 0.0, 0.0, 0.0],
        rotation_type="quat",
        quat_convention="wxyz"
    )
    print(f"EndEffectorPose.xyz_rot6d: {pose.xyz_rot6d}")

    # Test sequence conversion (like in dataloader)
    seq_endpose_7d = np.random.randn(10, 7)  # 10 timesteps
    seq_endpose_7d[:, 3:7] = seq_endpose_7d[:, 3:7] / np.linalg.norm(seq_endpose_7d[:, 3:7], axis=-1, keepdims=True)
    seq_endpose_9d = convert_endpose_7d_to_9d(seq_endpose_7d, quat_convention="wxyz")
    print(f"Sequence 7D shape: {seq_endpose_7d.shape} -> 9D shape: {seq_endpose_9d.shape}")

    print("\nAll rotation utility tests passed!")
