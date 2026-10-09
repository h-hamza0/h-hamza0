"""Render an animated SVG: a rotating 3D molecule while a message-passing GNN reads it out."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from html import escape
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors

from gnn import N_LAYERS, featurize, forward
from molecules import MOLECULES

ROOT = Path(__file__).resolve().parent.parent
WIDTH, HEIGHT = 820, 380
LOOP_SECONDS = 12
N_FRAMES = 96
TILT = np.radians(22)
MOLECULE_CENTRE = (205, 190)
MOLECULE_RADIUS = 165
GAUGE_RANGE = (-1.5, 4.5)
BASE_BOND, ACTIVE, BRIGHT = "#6e7681", "#2dd4bf", "#99f6e4"
CPK = {6: "#8b949e", 7: "#58a6ff", 8: "#f85149", 9: "#7ee787", 15: "#ffa657", 16: "#e3b341", 17: "#3fb950", 35: "#bc6c25", 53: "#bc8cff"}
SYMBOL_RADIUS = {6: 4.5}
STAGES = ["Atom embedding", "Message passing 1", "Message passing 2", "Message passing 3", "Readout"]
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def pick_molecule(index: int | None, name: str | None, today: dt.date) -> tuple[str, str]:
    if name:
        return next(m for m in MOLECULES if m[0].lower() == name.lower())
    return MOLECULES[(today.toordinal() if index is None else index) % len(MOLECULES)]


def embed_3d(smiles: str) -> tuple[Chem.Mol, np.ndarray]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        raise RuntimeError(f"3D embedding failed for {smiles}")
    if AllChem.MMFFOptimizeMolecule(mol, maxIters=500) < 0:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    heavy = Chem.RemoveHs(mol)
    coords = heavy.GetConformer().GetPositions()
    return heavy, coords - coords.mean(axis=0)


def rotated_frames(coords: np.ndarray) -> np.ndarray:
    radius = np.linalg.norm(coords, axis=1).max()
    scale = MOLECULE_RADIUS / max(radius, 1e-6)
    tilt = np.array([[1, 0, 0], [0, np.cos(TILT), -np.sin(TILT)], [0, np.sin(TILT), np.cos(TILT)]])
    frames = []
    for k in range(N_FRAMES + 1):
        a = 2 * np.pi * k / N_FRAMES
        spin = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        p = coords @ spin.T @ tilt.T * min(scale, 60.0)
        frames.append(p[:, :2] * [1, -1] + MOLECULE_CENTRE)
    return np.array(frames)


def mix(c0: str, c1: str, t: float) -> str:
    a, b = [np.array([int(c[i : i + 2], 16) for i in (1, 3, 5)]) for c in (c0, c1)]
    return "#" + "".join(f"{int(round(v)):02x}" for v in a + (b - a) * t)


def activation_levels(layers: list[np.ndarray]) -> list[np.ndarray]:
    levels = []
    for h in layers:
        norm = np.linalg.norm(h, axis=1)
        spread = norm.max() - norm.min()
        levels.append(0.15 + 0.85 * (norm - norm.min()) / spread if spread > 1e-9 else np.full(len(norm), 0.6))
    return levels


def stage_windows() -> list[tuple[float, float]]:
    steps = N_LAYERS + 1
    return [(k / (2 * steps), (k + 1) / (2 * steps)) for k in range(steps)] + [(0.5, 0.625)]


def travel_span(window: tuple[float, float]) -> tuple[float, float]:
    width = window[1] - window[0]
    return window[0] + 0.02 * width, window[0] + 0.82 * width


def arrival_span(window: tuple[float, float]) -> tuple[float, float]:
    width = window[1] - window[0]
    return window[0] + 0.6 * width, window[1] + 0.3 * width


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def timeline(spans: list[tuple[float, float, np.ndarray]], n: int, rise: float = 0.3, fall: float = 0.4, steps: int = 6):
    """keyTimes and, per entity, a level that eases up and down inside each (start, end, levels) span."""
    times, values = [0.0], [np.zeros(n)]

    def add(t: float, g: float, level: np.ndarray) -> None:
        times.append(max(t, times[-1] + 1e-4))
        values.append(level * g)

    for start, end, level in spans:
        width = end - start
        for k in range(steps + 1):
            add(start + rise * width * k / steps, smoothstep(k / steps), level)
        for k in range(1, steps + 1):
            add(end - fall * width + fall * width * k / steps, 1 - smoothstep(k / steps), level)
    times.append(max(1.0, times[-1] + 1e-4))
    values.append(np.zeros(n))
    return times, np.array(values)


def edge_messages(weights: dict[str, np.ndarray], layers: list[np.ndarray], edges: np.ndarray) -> list[np.ndarray]:
    """Magnitude of the message sent along every directed edge in each message-passing round."""
    src, dst = edges
    degree = np.bincount(dst, minlength=len(layers[0])).clip(min=1)
    levels = []
    for k in range(N_LAYERS):
        norm = np.linalg.norm(layers[k] @ weights[f"nbr_w{k}"], axis=1)[src] / degree[dst]
        spread = norm.max() - norm.min()
        levels.append(0.2 + 0.8 * (norm - norm.min()) / spread if spread > 1e-9 else np.full(len(norm), 0.6))
    return levels


def interpolate_frames(frames: np.ndarray, t: float) -> np.ndarray:
    position = (t % 1.0) * N_FRAMES
    lower = int(np.floor(position))
    weight = position - lower
    return frames[lower] * (1 - weight) + frames[min(lower + 1, N_FRAMES)] * weight


def particle_tracks(frames, edges, messages, windows, samples=24):
    """Per directed edge: keyTimes, positions, opacities and radii for a pulse that travels src -> dst."""
    tracks = []
    for e in range(edges.shape[1]):
        i, j = edges[0, e], edges[1, e]
        times, positions, opacity, radius = [0.0], [frames[0, i]], [0.0], [0.0]
        for k, level in enumerate(m[e] for m in messages):
            t0, t1 = travel_span(windows[k + 1])
            size = 3.0 + 4.0 * level
            for n in range(samples):
                s = n / (samples - 1)
                t = t0 + (t1 - t0) * s
                frame = interpolate_frames(frames, t)
                eased = smoothstep(s)
                envelope = smoothstep(s / 0.22) * smoothstep((1 - s) / 0.22)
                times.append(max(t, times[-1] + 1e-4))
                positions.append(frame[i] + (frame[j] - frame[i]) * eased)
                opacity.append(round(float(envelope * (0.6 + 0.4 * level)), 2))
                radius.append(round(float(size * (0.5 + 0.5 * envelope)), 1))
        times.append(max(1.0, times[-1] + 1e-4))
        positions.append(positions[-1])
        opacity.append(0.0)
        radius.append(0.0)
        tracks.append((times, positions, opacity, radius))
    return tracks


def fmt(values) -> str:
    return ";".join(f"{v:.2f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v) for v in values)


def bond_offsets(order: int) -> list[float]:
    return {1: [0.0], 2: [-2.2, 2.2], 3: [-3.4, 0.0, 3.4]}[order]


def bond_path(p: np.ndarray, q: np.ndarray, order: int) -> str:
    direction = q - p
    length = np.linalg.norm(direction) or 1.0
    normal = np.array([-direction[1], direction[0]]) / length
    return " ".join(
        f"M{(p + normal * o)[0]:.1f} {(p + normal * o)[1]:.1f}L{(q + normal * o)[0]:.1f} {(q + normal * o)[1]:.1f}"
        for o in bond_offsets(order)
    )


def build_svg(name: str, smiles: str) -> tuple[str, dict]:
    weights = dict(np.load(ROOT / "model" / "gnn_lipophilicity.npz"))
    metrics = json.loads((ROOT / "model" / "metrics.json").read_text())
    heavy, coords = embed_3d(smiles)
    x, edge_index = featurize(heavy)
    prediction, layers = forward(weights, x, edge_index)
    n = heavy.GetNumAtoms()
    frames = rotated_frames(coords)
    windows = stage_windows()
    atom_levels = activation_levels(layers)
    messages = edge_messages(weights, layers, edge_index)
    key_times, glow = timeline(
        [(*windows[0], atom_levels[0])]
        + [(*arrival_span(windows[k + 1]), atom_levels[k + 1]) for k in range(N_LAYERS)]
        + [(windows[-1][0] + 0.03, windows[-1][1], np.full(n, 0.6))],
        n,
    )
    n_bonds = edge_index.shape[1] // 2
    ends = edge_index[:, ::2]
    bond_mean = lambda level: (level[::2] + level[1::2]) / 2
    bond_times, bond_glow = timeline(
        [(*windows[0], (atom_levels[0][ends[0]] + atom_levels[0][ends[1]]) / 2)]
        + [(*travel_span(windows[k + 1]), bond_mean(messages[k])) for k in range(N_LAYERS)]
        + [(windows[-1][0] + 0.03, windows[-1][1], np.full(n_bonds, 0.35))],
        n_bonds,
    )
    bond_kt = ";".join(f"{t:.4f}" for t in bond_times)
    kek = Chem.Mol(heavy)
    Chem.Kekulize(kek, clearAromaticFlags=True)

    props = {
        "MW": Descriptors.MolWt(heavy),
        "cLogP": Crippen.MolLogP(heavy),
        "TPSA": rdMolDescriptors.CalcTPSA(heavy),
        "HBD": Lipinski.NumHDonors(heavy),
        "HBA": Lipinski.NumHAcceptors(heavy),
        "RotB": Lipinski.NumRotatableBonds(heavy),
    }
    kt = ";".join(f"{t:.4f}" for t in key_times)
    anim = f'dur="{LOOP_SECONDS}s" repeatCount="indefinite"'
    loop_kt = ";".join(f"{k / N_FRAMES:.4f}" for k in range(N_FRAMES + 1))

    bonds = []
    for b, bond in enumerate(kek.GetBonds()):
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        order = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}.get(bond.GetBondType(), 1)
        paths = ";".join(bond_path(frames[k, i], frames[k, j], order) for k in range(N_FRAMES + 1))
        colours = ";".join(mix(BASE_BOND, BRIGHT, bond_glow[s, b]) for s in range(len(bond_times)))
        widths = fmt([round(float(2.6 + 2.0 * bond_glow[s, b]), 2) for s in range(len(bond_times))])
        bonds.append(
            f'<path fill="none" stroke="{BASE_BOND}" stroke-width="2.6" stroke-linecap="round" d="{bond_path(frames[0, i], frames[0, j], order)}">'
            f'<animate attributeName="d" values="{paths}" keyTimes="{loop_kt}" {anim}/>'
            f'<animate attributeName="stroke" values="{colours}" keyTimes="{bond_kt}" {anim}/>'
            f'<animate attributeName="stroke-width" values="{widths}" keyTimes="{bond_kt}" {anim}/></path>'
        )

    particles = []
    for times, positions, opacity, radius in particle_tracks(frames, edge_index, messages, windows):
        particle_kt = ";".join(f"{t:.4f}" for t in times)
        moves = ";".join(f"{p[0]:.1f} {p[1]:.1f}" for p in positions)
        particles.append(
            f'<g transform="translate({positions[0][0]:.1f} {positions[0][1]:.1f})">'
            f'<animateTransform attributeName="transform" type="translate" values="{moves}" keyTimes="{particle_kt}" {anim}/>'
            f'<circle r="3" fill="{BRIGHT}" opacity="0">'
            f'<animate attributeName="opacity" values="{fmt(opacity)}" keyTimes="{particle_kt}" {anim}/>'
            f'<animate attributeName="r" values="{fmt([round(float(r), 1) for r in radius])}" keyTimes="{particle_kt}" {anim}/></circle></g>'
        )

    atoms = []
    for atom in heavy.GetAtoms():
        i, z = atom.GetIdx(), atom.GetAtomicNum()
        r = SYMBOL_RADIUS.get(z, 8.5)
        positions = ";".join(f"{frames[k, i, 0]:.1f} {frames[k, i, 1]:.1f}" for k in range(N_FRAMES + 1))
        label = "" if z == 6 else f'<text y="3.4" text-anchor="middle" font-size="10" font-weight="700" fill="#0d1117">{atom.GetSymbol()}</text>'
        atoms.append(
            f'<g transform="translate({frames[0, i, 0]:.1f} {frames[0, i, 1]:.1f})">'
            f'<animateTransform attributeName="transform" type="translate" values="{positions}" keyTimes="{loop_kt}" {anim}/>'
            f'<circle r="{r + 5}" fill="{ACTIVE}" opacity="0">'
            f'<animate attributeName="opacity" values="{fmt([round(float(0.4 + 0.6 * v) if v > 0 else 0.0, 2) for v in glow[:, i]])}" keyTimes="{kt}" {anim}/>'
            f'<animate attributeName="r" values="{fmt([round(float(r + 4 + 11 * v), 1) for v in glow[:, i]])}" keyTimes="{kt}" {anim}/></circle>'
            f'<circle r="{r}" fill="{CPK.get(z, "#c9d1d9")}"/>{label}</g>'
        )

    steps = []
    windows = stage_windows()
    for k, label in enumerate(STAGES):
        start, end = windows[k]
        values = fmt([0.35, 0.35, 1.0, 1.0, 0.35, 0.35])
        times = f"0;{max(start - 0.012, 0.001):.4f};{start + 0.012:.4f};{end - 0.012:.4f};{end + 0.012:.4f};1"
        steps.append(
            f'<g transform="translate(445 {132 + 24 * k})"><circle cx="5" cy="-4" r="4" fill="{ACTIVE}">'
            f'<animate attributeName="opacity" values="{values}" keyTimes="{times}" {anim}/></circle>'
            f'<text x="20" font-size="14" fill="#c9d1d9"><animate attributeName="opacity" values="{values}" keyTimes="{times}" {anim}/>{label}</text></g>'
        )

    lo, hi = GAUGE_RANGE
    fill = float(np.clip((prediction - lo) / (hi - lo), 0, 1)) * 345
    reveal = "0;0.5;0.56;0.93;1"
    shown = "0;0;1;1;0"
    pretty_smiles = smiles if len(smiles) <= 46 else smiles[:43] + "…"
    props_line = f"MW {props['MW']:.1f}  ·  cLogP {props['cLogP']:.2f}  ·  TPSA {props['TPSA']:.1f}  ·  HBD {props['HBD']}  ·  HBA {props['HBA']}  ·  RotB {props['RotB']}"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="t d" font-family="{FONT}">
<title id="t">Molecule of the day: {escape(name)}</title>
<desc id="d">A rotating 3D structure of {escape(name)} while a graph neural network predicts its lipophilicity (logD) as {prediction:.2f}.</desc>
<rect x="1" y="1" width="{WIDTH - 2}" height="{HEIGHT - 2}" rx="14" fill="#0d1117" stroke="#30363d" stroke-width="2"/>
<g>{"".join(bonds)}{"".join(particles)}{"".join(atoms)}</g>
<line x1="410" y1="26" x2="410" y2="{HEIGHT - 26}" stroke="#30363d"/>
<text x="445" y="46" font-size="11" letter-spacing="2" fill="#8b949e">MOLECULE OF THE DAY</text>
<text x="445" y="82" font-size="30" font-weight="700" fill="#f0f6fc">{escape(name)}</text>
<text x="445" y="104" font-size="11" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" fill="#8b949e">{escape(pretty_smiles)}</text>
{"".join(steps)}
<g opacity="0"><animate attributeName="opacity" values="{shown}" keyTimes="{reveal}" {anim}/>
<text x="445" y="262" font-size="12" fill="#8b949e">GNN-predicted lipophilicity, logD at pH 7.4</text>
<text x="445" y="304" font-size="38" font-weight="700" fill="{ACTIVE}">{prediction:+.2f}</text>
<rect x="445" y="316" width="345" height="7" rx="3.5" fill="#21262d"/>
<rect x="445" y="316" width="{fill:.1f}" height="7" rx="3.5" fill="{ACTIVE}"/></g>
<text x="445" y="346" font-size="11.5" fill="#c9d1d9">{props_line}</text>
<text x="445" y="364" font-size="10" fill="#6e7681">3-layer message-passing GNN · {metrics["n_molecules"]:,} molecules · held-out RMSE {metrics["test_rmse"]} (R² {metrics["test_r2"]}) · descriptors from RDKit</text>
</svg>
'''
    return svg, {"name": name, "smiles": smiles, "prediction": round(prediction, 3), **{k: round(float(v), 2) for k, v in props.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int)
    parser.add_argument("--name")
    parser.add_argument("--output", default=str(ROOT / "assets" / "molecule-of-the-day.svg"))
    args = parser.parse_args()
    name, smiles = pick_molecule(args.index, args.name, dt.datetime.now(dt.timezone.utc).date())
    svg, summary = build_svg(name, smiles)
    Path(args.output).write_text(svg)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
