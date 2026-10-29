#!/usr/bin/env python3
"""
extract_orca_inputs.py
Reads ccsdt-nomp2grad.yaml and writes one ORCA input file per step.
Requires PyYAML:  pip install pyyaml
"""
import yaml
from pathlib import Path

yaml_file = Path("ccsdt-nomp2grad.yaml")
data = yaml.safe_load(yaml_file.read_text())

for step in data["campaign_config"]["args"]:
    name = step["name"]
    route = step["route"].strip()
    # The basis/ECP blocks are stored inside the route string.
    # Split the route into the keyword line and the %-block section.
    lines = route.splitlines()
    keyword_line = lines[0].strip()
    block_lines = lines[1:]          # everything after the first line

    # Assemble the ORCA input
    inp_content = []
    inp_content.append(f"! {keyword_line}")
    # Add any %-blocks exactly as they appear (memory, basis, mdci, etc.)
    inp_content.extend(block_lines)
    # A simple title line is optional but harmless
    inp_content.append("* xyzfile 0 1 molecule.xyz")   # you will replace this later

    out_file = Path(f"{name}.inp")
    out_file.write_text("\n".join(inp_content) + "\n")
    print(f"Wrote {out_file}")
