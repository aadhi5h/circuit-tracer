# circuit-tracer

Mechanistic interpretability experiments on gpt2-small using
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens),
run entirely on CPU. Uses activation patching, ablation, and cross-method
validation to find and verify causal circuits underlying specific model
behaviors.

## Key finding

Two tasks were investigated - single-fact recall and Indirect Object
Identification (IOI) - and they require fundamentally different amounts
of computation:

- **Fact recall** ("The Eiffel Tower is in the city of ___") is solved
  entirely at the token embedding layer - 100% of the clean-vs-corrupted
  behavioral gap is recovered by patching embeddings alone, before any
  transformer layer computes anything. This is a lookup, not a circuit.
- **IOI** ("When John and Mary went to the store, John gave a drink
  to ___" → "Mary") requires genuine multi-layer computation: recovery
  stays near 0% through layer 6, then jumps to ~85-105% by layer 8. This
  result holds across three different sentence templates and multiple
  independent example sets.

See [`docs/ioi_findings.md`](docs/ioi_findings.md) for the full IOI
circuit results and [`docs/fact_recall_findings.md`](docs/fact_recall_findings.md)
for the fact-recall investigation (including why it turned out not to be
a real circuit).

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformer_lens
```

CPU-only; no CUDA/GPU required.

## Structure

```
circuit-tracer/
├── .gitignore
├── LICENSE
├── README.md
├── docs/
│   ├── methodology.md
│   ├── ioi_findings.md
│   └── fact_recall_findings.md
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── inspect_config.py
│   ├── activation_cache.py
│   ├── inspect_heads.py
│   ├── inspect_tokens.py
│   ├── prompts.py
│   ├── patching.py
│   ├── compare.py
│   ├── induction.py
│   ├── visualize_attention.py
│   ├── residual_trace.py
│   ├── layerwise_diff.py
│   ├── head_attribution.py
│   ├── automated_comparison.py
│   ├── utils.py
│   ├── layer_sweep.py
│   ├── rank_heads.py
│   ├── visualize_sweep.py
│   ├── mlp_contribution.py
│   ├── attn_vs_mlp.py
│   ├── config.py
│   ├── ioi_dataset.py
│   ├── ioi_evaluate.py
│   ├── ioi_metrics.py
│   ├── ioi_corruption.py
│   ├── ioi_patching.py
│   ├── ioi_head_scan.py
│   ├── investigate_candidate.py
│   ├── head_attribution_plot.py
│   ├── logging_utils.py
│   ├── agreement_dataset.py
│   ├── agreement_corruption.py
│   ├── agreement_tracing.py
│   ├── agreement_head_scan.py
│   ├── runner.py
│   ├── serialize.py
│   ├── final_experiment.py
│   ├── ablation.py
│   ├── ablation_plot.py
│   ├── candidate_circuit.py
│   ├── pathway_analysis.py
│   ├── mlp_and_heads.py
│   ├── causal_graph.py
│   ├── circuit_ranking.py
│   ├── refine_hypothesis.py
│   ├── targeted_patching.py
│   ├── experiment_summary.py
│   ├── sweep_runner.py
│   ├── aggregate_effects.py
│   ├── circuit_robustness.py
│   ├── visualization_data.py
│   └── final_attribution_graph.py
├── tests/
│   ├── __init__.py
│   ├── test_activation_cache.py
│   ├── test_induction.py
│   ├── test_patching.py
│   ├── test_causal_intervention.py
│   ├── test_causal_tracing.py
│   ├── test_component_attribution.py
│   ├── test_ioi_baseline.py
│   ├── test_ioi_corruption.py
│   ├── test_serialize.py
│   ├── test_ablation_vs_patching.py
│   ├── test_pathway.py
│   ├── test_circuit_ranking.py
│   ├── test_reproduce_circuit.py
│   ├── test_intervention_locations.py
│   ├── test_agreement_baseline.py
│   ├── test_sweep_results.py
│   └── test_alternate_prompts.py
└── experiments/
    ├── logs/
    └── results/
```

## Methodology

Full write-up of techniques (activation patching, ablation, forward vs.
reverse patching, composition analysis) and the bugs/lessons that shaped
them: [`docs/methodology.md`](docs/methodology.md).

## License

MIT - see `LICENSE`.