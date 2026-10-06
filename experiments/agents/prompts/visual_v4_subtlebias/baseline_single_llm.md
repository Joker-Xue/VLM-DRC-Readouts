You are inspecting a chip-layout region for DRC (design-rule-check) hotspot risk.

The image shows several feature-map panels of the SAME die area (top rows: full map with the
inspected region marked by a RED BOX; bottom rows: zoom into that region). Panels include
macro / cell-density, RUDY wire-demand, and routing-congestion overflow maps; the LAST panel
(CNN_prior) is a learned hotspot-prediction map from a CNN — strong evidence when it shows a bright
blob inside the red box, but it can miss subtle hotspots.

Intensities are normalized per patch, so judge LOCALIZED structure (sharp spikes, dense clusters,
bright bands inside the red box / zoom panels), not global brightness. First LOOK at the zoom panels
and describe what is actually inside the region; then rate the risk. Be cautious: only assign a high
probability when the evidence is genuinely clear, not merely suggestive.

Answer with JSON only — observation must be ONE short sentence (a plain string, max 25 words, NOT
per-panel):
{"observation": "<one sentence>", "p_hotspot": <single number 0..1>}
