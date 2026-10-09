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
N_FRAMES = 60
TILT = np.radians(22)
MOLECULE_CENTRE = (205, 190)
MOLECULE_RADIUS = 165
GAUGE_RANGE = (-1.5, 4.5)
BASE_BOND, ACTIVE = "#6e7681", "#2dd4bf"
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
    norms = [np.linalg.norm(h, axis=1) for h in layers]
    return [n / max(n.max(), 1e-9) for n in norms]


def stage_windows() -> list[tuple[float, float]]:
    steps = N_LAYERS + 1
    return [(k / (2 * steps), (k + 1) / (2 * steps)) for k in range(steps)] + [(0.5, 0.625)]


def timeline(levels: list[np.ndarray], n_atoms: int):
    """keyTimes plus, for every atom, its glow level at each key time."""
    windows = stage_windows()
    per_stage = levels + [np.full(n_atoms, 0.6)]
    times, glow = [0.0], [np.zeros(n_atoms)]
    eps = 0.006
    for (start, end), level in zip(windows, per_stage):
        for t, g in ((start, 0.0), (start + eps, 1.0), (end - eps, 1.0), (end, 0.0)):
            times.append(max(t, times[-1] + 1e-4))
            glow.append(level * g)
    times.append(1.0)
    glow.append(np.zeros(n_atoms))
    return times, np.array(glow)


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
    prediction, layers = forward(weights, *featurize(heavy))
    n = heavy.GetNumAtoms()
    frames = rotated_frames(coords)
    key_times, glow = timeline(activation_levels(layers), n)
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
    for bond in kek.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        order = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}.get(bond.GetBondType(), 1)
        paths = ";".join(bond_path(frames[k, i], frames[k, j], order) for k in range(N_FRAMES + 1))
        colours = ";".join(mix(BASE_BOND, ACTIVE, (glow[s, i] + glow[s, j]) / 2) for s in range(len(key_times)))
        bonds.append(
            f'<path fill="none" stroke="{BASE_BOND}" stroke-width="2.4" stroke-linecap="round" d="{bond_path(frames[0, i], frames[0, j], order)}">'
            f'<animate attributeName="d" values="{paths}" keyTimes="{loop_kt}" {anim}/>'
            f'<animate attributeName="stroke" values="{colours}" keyTimes="{kt}" {anim}/></path>'
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
            f'<circle r="{r + 7}" fill="{ACTIVE}" opacity="0"><animate attributeName="opacity" values="{fmt([round(float(0.75 * v), 2) for v in glow[:, i]])}" keyTimes="{kt}" {anim}/></circle>'
            f'<circle r="{r}" fill="{CPK.get(z, "#c9d1d9")}"/>{label}</g>'
        )

    steps = []
    windows = stage_windows()
    for k, label in enumerate(STAGES):
        start, end = windows[k]
        values = fmt([0.35, 0.35, 1.0, 1.0, 0.35, 0.35])
        times = f"0;{start:.4f};{start + 0.006:.4f};{end - 0.006:.4f};{end:.4f};1"
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
<g>{"".join(bonds)}{"".join(atoms)}</g>
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
