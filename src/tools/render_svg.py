"""Server-side 2D projection of a docking scene, as SVG.

The dashboard draws the pose interactively on a canvas; this renders the same scene
to a static SVG so it can be printed, embedded in a report, or screenshotted by a
headless browser that has no JavaScript engine.
"""

import math

ELEMENT_COLOURS = {"C": "#8fa3bf", "N": "#6fa8ff", "O": "#ff7b7b", "S": "#f0c674",
                   "F": "#8ce99a", "CL": "#8ce99a", "BR": "#b06a4a"}
ELEMENT_RADII = {"C": 3.0, "N": 3.0, "O": 2.9, "S": 3.6, "F": 2.7, "CL": 3.4, "BR": 3.7}


def _rotate(point, yaw, pitch):
    x, y, z = point
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    x, z = x * cos_y - z * sin_y, x * sin_y + z * cos_y
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    y, z = y * cos_p - z * sin_p, y * sin_p + z * cos_p
    return x, y, z


def render(scene: dict, width: int = 860, height: int = 620,
           yaw: float = -0.55, pitch: float = 0.32) -> str:
    centre = scene["centre"]
    points = []
    for atom in scene["protein"]:
        points.append(("protein", atom, (atom["x"] - centre[0], atom["y"] - centre[1],
                                         atom["z"] - centre[2])))
    for atom in scene["ligand"]:
        points.append(("ligand", atom, (atom["x"] - centre[0], atom["y"] - centre[1],
                                        atom["z"] - centre[2])))

    rotated = [(kind, atom, _rotate(xyz, yaw, pitch)) for kind, atom, xyz in points]
    span = max(max(abs(p[2][0]), abs(p[2][1])) for p in rotated) or 1.0
    scale = min(width, height) * 0.42 / span
    depth = [p[2][2] for p in rotated]
    low, high = min(depth), max(depth)

    def project(xyz):
        return width / 2 + xyz[0] * scale, height / 2 - xyz[1] * scale

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'style="background:#0a1420;border-radius:10px">',
             f'<rect width="{width}" height="{height}" fill="#0a1420"/>']

    # --- protein backbone: a continuous trace through C-alpha atoms
    backbone = scene.get("backbone") or []
    if backbone:
        path_parts = []
        for index, point in enumerate(backbone):
            px, py = project(_rotate((point["p"][0] - centre[0], point["p"][1] - centre[1],
                                      point["p"][2] - centre[2]), yaw, pitch))
            path_parts.append(f'{"M" if index == 0 or point["gap"] else "L"}{px:.1f} {py:.1f}')
        parts.append(f'<path d="{" ".join(path_parts)}" fill="none" stroke="#7f97b3" '
                     f'stroke-width="4.5" stroke-linecap="round" opacity="0.85"/>')

    # --- side chains of the interacting residues, so the pocket is visible
    for stick in scene.get("sticks") or []:
        a = project(_rotate((stick["a"][0] - centre[0], stick["a"][1] - centre[1],
                             stick["a"][2] - centre[2]), yaw, pitch))
        b = project(_rotate((stick["b"][0] - centre[0], stick["b"][1] - centre[1],
                             stick["b"][2] - centre[2]), yaw, pitch))
        parts.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
                     f'stroke="#3ddc97" stroke-width="2.2" stroke-linecap="round" opacity="0.75"/>')

    for kind, atom, xyz in sorted(rotated, key=lambda item: item[2][2]):
        px, py = project(xyz)
        fade = 0.25 + 0.6 * ((xyz[2] - low) / (high - low or 1))
        if kind == "protein":
            colour = "#5f7f9e" if atom["kind"] == "protein" else "#3ddc97"
            radius = 2.2 if atom["kind"] == "protein" else 3.4
            opacity = fade * (0.55 if atom["kind"] == "protein" else 1.0)
        else:
            colour = ELEMENT_COLOURS.get(atom["e"], "#ffd479")
            radius = ELEMENT_RADII.get(atom["e"], 3.0) + 1.4
            opacity = 1.0
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius:.1f}" '
                     f'fill="{colour}" opacity="{opacity:.2f}"/>')

    ligand_xy = [project(_rotate((a["x"] - centre[0], a["y"] - centre[1],
                                  a["z"] - centre[2]), yaw, pitch))
                 for a in scene["ligand"]]
    for i, j in scene["bonds"]:
        if i < len(ligand_xy) and j < len(ligand_xy):
            (x1, y1), (x2, y2) = ligand_xy[i], ligand_xy[j]
            parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                         f'stroke="#ffb454" stroke-width="4.2" stroke-linecap="round"/>')

    for bond in scene["hbonds"]:
        start = project(_rotate((bond["from"][0] - centre[0], bond["from"][1] - centre[1],
                                 bond["from"][2] - centre[2]), yaw, pitch))
        end = project(_rotate((bond["to"][0] - centre[0], bond["to"][1] - centre[1],
                               bond["to"][2] - centre[2]), yaw, pitch))
        parts.append(f'<line x1="{start[0]:.1f}" y1="{start[1]:.1f}" x2="{end[0]:.1f}" '
                     f'y2="{end[1]:.1f}" stroke="#ff8fa3" stroke-width="1.6" '
                     f'stroke-dasharray="4 4"/>')

    placed = []
    for label in scene["labels"][:12]:
        px, py = project(_rotate((label["x"] - centre[0], label["y"] - centre[1],
                                  label["z"] - centre[2]), yaw, pitch))
        if any(abs(px - x) < 62 and abs(py - y) < 18 for x, y in placed):
            continue                      # avoid overlapping labels
        placed.append((px, py))
        parts.append(f'<text x="{px + 6:.1f}" y="{py - 5:.1f}" fill="#9fd8c0" '
                     f'font-family="monospace" font-size="11">{label["text"]}</text>')

    legend = [("#ffb454", "ligand (docked pose)"), ("#3ddc97", "interacting residues"),
              ("#7f97b3", "protein backbone"), ("#ff8fa3", "hydrogen bonds")]
    for index, (colour, text) in enumerate(legend):
        y = 22 + index * 18
        parts.append(f'<rect x="16" y="{y - 9}" width="10" height="10" rx="2" fill="{colour}"/>'
                     f'<text x="32" y="{y}" fill="#9db2c9" font-size="11" '
                     f'font-family="monospace">{text}</text>')
    parts.append(f'<text x="16" y="{height - 12}" fill="#5f7f9e" font-size="11" '
                 f'font-family="monospace">protein {scene["counts"]["protein_atoms"]} atoms · '
                 f'ligand {scene["counts"]["ligand_atoms"]} · '
                 f'{scene["counts"]["hbonds"]} H-bonds · pose 1 of 9</text>')
    parts.append("</svg>")
    return "".join(parts)
