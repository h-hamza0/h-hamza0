import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import molecule_of_the_day as widget  # noqa: E402
from gnn import featurize, forward  # noqa: E402
from molecules import MOLECULES  # noqa: E402
from rdkit import Chem  # noqa: E402


def test_catalogue_is_valid_and_unique():
    names = [n for n, _ in MOLECULES]
    assert len(set(names)) == len(names)
    for name, smiles in MOLECULES:
        assert Chem.MolFromSmiles(smiles) is not None, name


def test_numpy_forward_is_finite_and_permutation_invariant():
    weights = dict(np.load(ROOT / "model" / "gnn_lipophilicity.npz"))
    mol = Chem.MolFromSmiles(MOLECULES[0][1])
    x, edges = featurize(mol)
    value, layers = forward(weights, x, edges)
    assert np.isfinite(value) and len(layers) == 4

    order = np.random.default_rng(0).permutation(len(x))
    inverse = np.argsort(order)
    permuted, _ = forward(weights, x[order], inverse[edges])
    assert permuted == pytest.approx(value, abs=1e-4)


@pytest.mark.parametrize("name", [n for n, _ in MOLECULES])
def test_svg_renders_for_every_molecule(name, tmp_path):
    svg, summary = widget.build_svg(*widget.pick_molecule(None, name, None))
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert -4 < summary["prediction"] < 7


def test_daily_selection_cycles():
    import datetime as dt

    first = widget.pick_molecule(None, None, dt.date(2026, 1, 1))
    later = widget.pick_molecule(None, None, dt.date(2026, 1, 1) + dt.timedelta(days=len(MOLECULES)))
    assert first == later
