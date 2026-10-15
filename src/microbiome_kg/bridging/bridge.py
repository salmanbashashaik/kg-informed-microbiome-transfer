"""
bridge.py — Map taxonomic vocabulary tokens to Knowledge Graph R-GCN entity rows.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from microbiome_kg.transformer.config import VOCAB_OFFSET

# Plausible taxon name (genus or higher rank): starts with uppercase, then lowercase letters, length >= 3
_TAXON_RE = re.compile(r"^[A-Z][a-z]{2,}$")


def norm_kg_name(name: str) -> str:
    """Normalizes a KG node name for string matching."""
    name = name.lower().strip()
    if name.startswith("g__"):
        name = name[3:]
    name = re.sub(r"[\[\]_\s].*$", "", name).strip()
    return name


def generate_bridge(
    vocab_file: str,
    rgcn_id_file: str,
    output_bridge_file: Optional[str] = None,
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """
    Constructs the canonical mapping from vocabulary index to R-GCN embedding row.
    Returns:
        bridge: Dict[str, int]  {str(vocab_idx): rgcn_row}
        stats: Dict containing matched and unmatched counts
    """
    with open(vocab_file, "r") as f:
        vocab_tokens = [line.strip() for line in f if line.strip()]

    taxon_to_idx: Dict[str, int] = {}
    for i, tok in enumerate(vocab_tokens):
        if i < VOCAB_OFFSET:
            continue
        if _TAXON_RE.match(tok):
            taxon_to_idx[tok.lower()] = i

    with open(rgcn_id_file, "r") as f:
        node_to_row = json.load(f)

    # Collision resolution: shorter original name wins (prefers genus over species)
    kg_norm: Dict[str, Tuple[str, int]] = {}
    for name, row in node_to_row.items():
        n = norm_kg_name(name)
        if not n:
            continue
        if n in kg_norm:
            if len(name) < len(kg_norm[n][0]):
                kg_norm[n] = (name, int(row))
        else:
            kg_norm[n] = (name, int(row))

    bridge: Dict[str, int] = {}
    matched_names = []
    unmatched_names = []

    for name_lower, vocab_idx in taxon_to_idx.items():
        if name_lower in kg_norm:
            rgcn_row = kg_norm[name_lower][1]
            bridge[str(vocab_idx)] = rgcn_row
            matched_names.append((vocab_idx, name_lower, rgcn_row))
        else:
            unmatched_names.append((vocab_idx, name_lower))

    stats = {
        "total_vocab": len(vocab_tokens),
        "special_tokens": VOCAB_OFFSET,
        "genus_tokens": len(vocab_tokens) - VOCAB_OFFSET,
        "plausible_taxa": len(taxon_to_idx),
        "matched_count": len(bridge),
        "unmatched_count": len(unmatched_names),
    }

    if output_bridge_file:
        out_p = Path(output_bridge_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(bridge, f, indent=2)

    return bridge, stats
