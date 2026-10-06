You are inspecting a chip-layout region for DRC (design-rule-check) hotspot risk.

The image shows several feature-map panels of the SAME die area (top rows: full map with the
inspected region marked by a RED BOX; bottom rows: zoom into that region). Panels include
macro / cell-density, RUDY wire-demand, and routing-congestion overflow maps.

Intensities are normalized per patch, so judge LOCALIZED structure (sharp spikes, dense clusters,
bright bands inside the red box / zoom panels), not global brightness. First LOOK at the zoom panels
and describe what is actually inside the region; then rate the risk.

Answer with JSON only — observation must be ONE short sentence (a plain string, max 25 words, NOT
per-panel):
{"observation": "<one sentence>", "p_hotspot": <single number 0..1>}
