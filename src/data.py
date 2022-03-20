from math import pi
import pandas as pd
from dataclasses import dataclass

coordinate_names = [
    # Pelvis rotation
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
    # Pelvis position
    "pelvis_tx",
    "pelvis_ty",
    "pelvis_tz",
    # Right hip angles
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    # Right knee and ankle angles
    "knee_angle_r",
    "ankle_angle_r",
    # Left hip angles
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    # Left knee and ankle angles
    "knee_angle_l",
    "ankle_angle_l",
]

angular_coordinate_names = [
    # Pelvis rotation
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
    # Right hip angles
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    # Right knee and ankle angles
    "knee_angle_r",
    "ankle_angle_r",
    # Left hip angles
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    # Left knee and ankle angles
    "knee_angle_l",
    "ankle_angle_l",
]


class TrainingData:
    def __init__(self, path, deg_to_rad=True, tempo=1, start_time=0):
        self.deg_to_rad = deg_to_rad
        self.tempo = tempo
        self.start_time = start_time

        self.data = pd.read_csv(path, sep=",")
        self.data.sort_values("time")
        self.data.drop_duplicates("time")

        if deg_to_rad:
            self.data[angular_coordinate_names] *= pi / 180.0

        self.guessed_delta_time = (
            self.data["time"].iloc[-1] - self.data["time"][0]
        ) / (self.data.shape[0] - 1)

    def _get_last_index_before(self, time):
        """Returns the index of the column at time, or immediately before time."""
        guessed_index = round(time / self.guessed_delta_time)

        if guessed_index >= self.data.shape[0]:
            return None

        guess = self.data["time"][guessed_index]
        if guess > time:
            while guess > time:
                if guessed_index == 0:
                    return None
                guessed_index -= 1
                guess = self.data["time"][guessed_index]
            return guessed_index
        elif guess < time:
            while guess < time:
                guessed_index += 1
                if guessed_index >= self.data.shape[0]:
                    return guessed_index

                guess = self.data["time"][guessed_index]

            return guessed_index - 1
        else:
            return guessed_index

    def get_row(self, time):
        time *= self.tempo
        time += self.start_time

        idx = self._get_last_index_before(time)

        if idx == None or idx + 1 >= self.data.shape[0]:
            return None

        a = self.data.iloc[idx, :]
        b = self.data.iloc[idx + 1, :]

        dt = b["time"] - a["time"]
        ifac = (time - a["time"]) / dt

        return DataRow.interpolate(a, b, ifac, dt)

@dataclass
class DataRow:
    pelvis_tilt: float
    pelvis_list: float
    pelvis_rotation: float
    pelvis_tx: float
    pelvis_ty: float
    pelvis_tz: float
    hip_flexion_r: float
    hip_adduction_r: float
    hip_rotation_r: float
    knee_angle_r: float
    ankle_angle_r: float
    hip_flexion_l: float
    hip_adduction_l: float
    hip_rotation_l: float
    knee_angle_l: float
    ankle_angle_l: float

    d_pelvis_tilt: float
    d_pelvis_list: float
    d_pelvis_rotation: float
    d_pelvis_tx: float
    d_pelvis_ty: float
    d_pelvis_tz: float
    d_hip_flexion_r: float
    d_hip_adduction_r: float
    d_hip_rotation_r: float
    d_knee_angle_r: float
    d_ankle_angle_r: float
    d_hip_flexion_l: float
    d_hip_adduction_l: float
    d_hip_rotation_l: float
    d_knee_angle_l: float
    d_ankle_angle_l: float

    def __init__(self):
        pass

    @staticmethod
    def interpolate(row, row_next, interpolation_factor, delta_time) -> "DataRow":
        r = DataRow()
        a = 1.0 - interpolation_factor
        b = interpolation_factor

        # fmt: off
        r.pelvis_tilt       = row["pelvis_tilt"]     * a   + row_next["pelvis_tilt"]     * b
        r.pelvis_list       = row["pelvis_list"]     * a   + row_next["pelvis_list"]     * b
        r.pelvis_rotation   = row["pelvis_rotation"] * a   + row_next["pelvis_rotation"] * b
        r.pelvis_tx         = row["pelvis_tx"]       * a   + row_next["pelvis_tx"]       * b
        r.pelvis_ty         = row["pelvis_ty"]       * a   + row_next["pelvis_ty"]       * b
        r.pelvis_tz         = row["pelvis_tz"]       * a   + row_next["pelvis_tz"]       * b
        r.hip_flexion_r     = row["hip_flexion_r"]   * a   + row_next["hip_flexion_r"]   * b
        r.hip_adduction_r   = row["hip_adduction_r"] * a   + row_next["hip_adduction_r"] * b
        r.hip_rotation_r    = row["hip_rotation_r"]  * a   + row_next["hip_rotation_r"]  * b
        r.knee_angle_r      = row["knee_angle_r"]    * a   + row_next["knee_angle_r"]    * b
        r.ankle_angle_r     = row["ankle_angle_r"]   * a   + row_next["ankle_angle_r"]   * b
        r.hip_flexion_l     = row["hip_flexion_l"]   * a   + row_next["hip_flexion_l"]   * b
        r.hip_adduction_l   = row["hip_adduction_l"] * a   + row_next["hip_adduction_l"] * b
        r.hip_rotation_l    = row["hip_rotation_l"]  * a   + row_next["hip_rotation_l"]  * b
        r.knee_angle_l      = row["knee_angle_l"]    * a   + row_next["knee_angle_l"]    * b
        r.ankle_angle_l     = row["ankle_angle_l"]   * a   + row_next["ankle_angle_l"]   * b

        r.d_pelvis_tilt     = (row_next["pelvis_tilt"]     - row["pelvis_tilt"])     / delta_time
        r.d_pelvis_list     = (row_next["pelvis_list"]     - row["pelvis_list"])     / delta_time
        r.d_pelvis_rotation = (row_next["pelvis_rotation"] - row["pelvis_rotation"]) / delta_time
        r.d_pelvis_tx       = (row_next["pelvis_tx"]       - row["pelvis_tx"])       / delta_time
        r.d_pelvis_ty       = (row_next["pelvis_ty"]       - row["pelvis_ty"])       / delta_time
        r.d_pelvis_tz       = (row_next["pelvis_tz"]       - row["pelvis_tz"])       / delta_time
        r.d_hip_flexion_r   = (row_next["hip_flexion_r"]   - row["hip_flexion_r"])   / delta_time
        r.d_hip_adduction_r = (row_next["hip_adduction_r"] - row["hip_adduction_r"]) / delta_time
        r.d_hip_rotation_r  = (row_next["hip_rotation_r"]  - row["hip_rotation_r"])  / delta_time
        r.d_knee_angle_r    = (row_next["knee_angle_r"]    - row["knee_angle_r"])    / delta_time
        r.d_ankle_angle_r   = (row_next["ankle_angle_r"]   - row["ankle_angle_r"])   / delta_time
        r.d_hip_flexion_l   = (row_next["hip_flexion_l"]   - row["hip_flexion_l"])   / delta_time
        r.d_hip_adduction_l = (row_next["hip_adduction_l"] - row["hip_adduction_l"]) / delta_time
        r.d_hip_rotation_l  = (row_next["hip_rotation_l"]  - row["hip_rotation_l"])  / delta_time
        r.d_knee_angle_l    = (row_next["knee_angle_l"]    - row["knee_angle_l"])    / delta_time
        r.d_ankle_angle_l   = (row_next["ankle_angle_l"]   - row["ankle_angle_l"])   / delta_time
        # fmt: on

        return r

    def __str__(self):
        return "DataRow({}, {})".format(
            [
                self.pelvis_tilt,
                self.pelvis_list,
                self.pelvis_rotation,
                self.pelvis_tx,
                self.pelvis_ty,
                self.pelvis_tz,
                self.hip_flexion_r,
                self.hip_adduction_r,
                self.hip_rotation_r,
                self.knee_angle_r,
                self.ankle_angle_r,
                self.hip_flexion_l,
                self.hip_adduction_l,
                self.hip_rotation_l,
                self.knee_angle_l,
                self.ankle_angle_l,
            ],
            [
                self.d_pelvis_tilt,
                self.d_pelvis_list,
                self.d_pelvis_rotation,
                self.d_pelvis_tx,
                self.d_pelvis_ty,
                self.d_pelvis_tz,
                self.d_hip_flexion_r,
                self.d_hip_adduction_r,
                self.d_hip_rotation_r,
                self.d_knee_angle_r,
                self.d_ankle_angle_r,
                self.d_hip_flexion_l,
                self.d_hip_adduction_l,
                self.d_hip_rotation_l,
                self.d_knee_angle_l,
                self.d_ankle_angle_l,
            ],
        )

    def as_array(self):
        return [
            self.pelvis_tilt,
            self.pelvis_list,
            self.pelvis_rotation,
            self.pelvis_tx,
            self.pelvis_ty,
            self.pelvis_tz,
            self.hip_flexion_r,
            self.hip_adduction_r,
            self.hip_rotation_r,
            self.knee_angle_r,
            self.ankle_angle_r,
            self.hip_flexion_l,
            self.hip_adduction_l,
            self.hip_rotation_l,
            self.knee_angle_l,
            self.ankle_angle_l,
            self.d_pelvis_tilt,
            self.d_pelvis_list,
            self.d_pelvis_rotation,
            self.d_pelvis_tx,
            self.d_pelvis_ty,
            self.d_pelvis_tz,
            self.d_hip_flexion_r,
            self.d_hip_adduction_r,
            self.d_hip_rotation_r,
            self.d_knee_angle_r,
            self.d_ankle_angle_r,
            self.d_hip_flexion_l,
            self.d_hip_adduction_l,
            self.d_hip_rotation_l,
            self.d_knee_angle_l,
            self.d_ankle_angle_l,
        ]
