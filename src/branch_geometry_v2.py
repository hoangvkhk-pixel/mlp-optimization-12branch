from __future__ import annotations

import math

import numpy as np
import pandas as pd


def decode_scheme(value: float) -> int:
    if value <= 1.0 / 3.0:
        return 1
    if value >= 2.0 / 3.0:
        return 3
    return 2


def normalized_scheme_value(value: float | int) -> float:
    if value in (1, 2, 3):
        return {1: 0.0, 2: 0.5, 3: 1.0}[int(value)]
    return float(value)


def build_geom_from_row(row: dict[str, float] | pd.Series, ac_module):
    scheme_fuse = decode_scheme(float(row["scheme_fuse"]))
    scheme_vertical = decode_scheme(float(row["scheme_vertical"]))
    a_dihedral_mag = float(row["a_dihedral_mag"])
    a_s_rel = float(row["a_S_rel"])
    f_area = float(row["S_ref"]) * (1.0 - a_s_rel)
    a_area = float(row["S_ref"]) * a_s_rel

    if a_s_rel <= 0.5:
        aft_dihedral = {1: a_dihedral_mag, 2: 0.0, 3: -a_dihedral_mag}[scheme_vertical]
        a_aspect = float(row["a_aspect"])
        if abs(aft_dihedral) > 1e-12 and a_aspect > 7.0:
            a_aspect = 7.0
        f_geo = ac_module.input_lift_surface_data(
            float(row["f_aspect"]),
            float(row["f_sweep"]),
            float(row["f_taper"]),
            float(row["f_twist"]),
            0.0,
            0.0,
            f_area,
        )
        a_geo = ac_module.input_lift_surface_data(
            a_aspect,
            float(row["a_sweep"]),
            float(row["a_taper"]),
            float(row["a_twist"]),
            aft_dihedral,
            0.0,
            a_area,
            float(row["a_x_loc"]),
        )
        if abs(aft_dihedral) > 1e-12:
            v_aspect = 0.0
            v_area = 0.0
        else:
            v_aspect = float(row["v_aspect"])
            v_area = float(row["S_ref"]) * float(row["v_S_rel"])
    else:
        f_geo = ac_module.input_lift_surface_data(
            float(row["f_aspect"]),
            0.0,
            float(row["f_taper"]),
            float(row["f_twist"]),
            0.0,
            0.0,
            f_area,
        )
        a_geo = ac_module.input_lift_surface_data(
            float(row["a_aspect"]),
            float(row["a_sweep"]),
            float(row["a_taper"]),
            float(row["a_twist"]),
            a_dihedral_mag,
            0.0,
            a_area,
            float(row["a_x_loc"]),
        )
        v_aspect = float(row["v_aspect"])
        v_area = float(row["S_ref"]) * float(row["v_S_rel"])

    v_geo = ac_module.input_lift_surface_data(
        v_aspect,
        15.0,
        2.0,
        0.0,
        90.0,
        0.0,
        v_area,
        float(row["a_x_loc"]),
    )

    f_span, f_root_chord, _ = ac_module.lift_surface_def(f_geo)
    a_span, a_root_chord, a_tip_chord = ac_module.lift_surface_def(a_geo)
    _, v_root_chord, _ = ac_module.lift_surface_def(v_geo)
    distance_two_fuse = min(float(a_span), float(f_span))
    ref_geo = f_geo if f_area >= a_area else a_geo
    mac_ref = float(ac_module.ref_dim_lift_surface(ref_geo))
    if not math.isfinite(mac_ref) or mac_ref <= 0:
        raise ValueError("Invalid reference MAC")
    a_loc = float(row["a_x_loc"]) * mac_ref
    v_loc = a_loc

    if scheme_fuse == 1:
        fuse_diameter = 0.76 / 2.5
    elif scheme_fuse == 2:
        fuse_diameter = 0.76 / 1.5
    else:
        fuse_diameter = 0.76
    nose_f_aspect = 2.0

    l_nose = nose_f_aspect * fuse_diameter
    if scheme_fuse == 3:
        main_x = a_loc if a_s_rel > 0.5 else 0.0
        main_root = a_root_chord if a_s_rel > 0.5 else f_root_chord
        l_center = main_root
        l_tail = 2.0 * fuse_diameter
        fuse_x_loc = main_x - l_nose
        if a_s_rel <= 0.5 and main_x + l_center + l_tail >= a_loc:
            raise ValueError("Short pod overlaps the aft surface")
    else:
        fuse_x_loc = -l_nose
        y_attach = distance_two_fuse / 2.0 if scheme_fuse == 1 else 0.0
        local_chord = (a_root_chord + (a_tip_chord - a_root_chord) *
                       (2.0 * y_attach / a_span)) if a_span > 0 else 0.0
        aft_te = a_loc + y_attach * math.tan(math.radians(float(a_geo[1]))) + local_chord
        fin_te = v_loc + v_root_chord if v_area > 0 else aft_te
        body_end = max(aft_te, fin_te)
        l_tail = 2.0 * fuse_diameter
        l_center = body_end - l_tail
        if l_center <= 0:
            raise ValueError("Long body has no cylindrical segment")
    center_f_aspect = l_center / fuse_diameter
    tail_f_aspect = l_tail / fuse_diameter
    fuse_geo = ac_module.input_body_data(
        nose_f_aspect,
        center_f_aspect,
        tail_f_aspect,
        fuse_diameter,
        fuse_diameter,
        fuse_x_loc,
    )
    return np.asarray(f_geo), np.asarray(a_geo), np.asarray(v_geo), np.asarray(fuse_geo), scheme_fuse
