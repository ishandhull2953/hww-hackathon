"""Print the branch layout of dataWW_d1.root (tree_event) from uproot.

Run from anywhere:  python minihep/inspect_branches.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import uproot

ROOT_FILE = Path(__file__).resolve().parents[1] / "dataWW_d1.root"


def main() -> None:
    f = uproot.open(ROOT_FILE)
    tree = f["tree_event"]
    print(f"file: {ROOT_FILE}")
    print(f"tree: 'tree_event'  entries: {tree.num_entries:,}")
    print(f"branches: {len(tree.keys())}")
    print()
    print(f"{'branch':<18} {'type':<10} note")
    print("-" * 70)
    arr = tree.arrays(library="np")
    for name, br in tree.items():
        note = ""
        if name == "lep_n":
            note = "no. of leptons (2..5); events with != 2 are dropped"
        elif name.startswith("lep_type_"):
            note = "PDG flavour: 11 = electron, 13 = muon"
        elif name.startswith("lep_charge_"):
            note = "+-1"
        elif name.startswith("lep_E_"):
            note = "ENERGY IN MeV (divide by 1000 for GeV)"
        elif name == "jet_n":
            note = "no. of jets in the event (can exceed 2; only 2 slots stored)"
        elif name.startswith("jet_E_"):
            note = "ENERGY IN MeV (divide by 1000 for GeV)"
        elif name.startswith("jet_"):
            note = "-7.0 filler when the jet is not present"
        elif name == "label":
            note = "1 = signal (WW), 0 = background"
        elif name == "mcWeight":
            note = "MC event weight (can be negative)"
        elif name == "met_et":
            note = "MET magnitude, GeV"
        elif name == "met_phi":
            note = "MET azimuth, rad"
        elif name == "index":
            note = "row number in this file"
        elif name == "eventNumber":
            note = "ATLAS event number"
        elif name == "runNumber":
            note = "ATLAS run number (284500)"
        elif name == "channelNumber":
            note = "sample channel id (345323/345324/363492)"
        print(f"{name:<18} {br.typename:<10} {note}")


if __name__ == "__main__":
    main()
