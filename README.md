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
│   ├── model.py                    # load gpt2-small
│   ├── inspect_config.py           # print model architecture
│   ├── activation_cache.py         # run_with_cache wrapper
│   ├── inspect_heads.py            # per-head attention pattern extraction
│   ├── inspect_tokens.py           # token-level residual inspection
│   ├── prompts.py                  # Paris/London clean/corrupted pair
│   ├── patching.py                 # core single-position patch_activation
│   ├── compare.py                  # clean/corrupted/patched logit diff
│   ├── induction.py                # induction head repeated-sequence test
│   ├── visualize_attention.py      # induction head scoring
│   ├── residual_trace.py           # decompose_resid by component
│   ├── layerwise_diff.py           # clean vs corrupted resid diff per layer
│   ├── head_attribution.py         # direct logit attribution per head
│   ├── automated_comparison.py     # per-layer patching sweep (Paris/London)
│   ├── utils.py                    # shared: get_answer_logit_diff, full_hook_name
│   ├── layer_sweep.py              # structured layer sweep
│   ├── rank_heads.py               # per-head patching, all 144 heads
│   ├── visualize_sweep.py          # text bar chart, layer sweep
│   ├── mlp_contribution.py         # per-layer MLP direct attribution
│   ├── attn_vs_mlp.py              # head vs MLP patching comparison
│   ├── config.py                   # ExperimentConfig dataclass
│   ├── ioi_dataset.py              # IOI 90-example dataset builder
│   ├── ioi_evaluate.py             # full-dataset IOI baseline eval
│   ├── ioi_metrics.py              # per-template accuracy/stats
│   ├── ioi_corruption.py           # ABC-corruption for IOI
│   ├── ioi_patching.py             # layer patching on IOI
│   ├── ioi_head_scan.py            # per-head patching on IOI, averaged
│   ├── investigate_candidate.py    # forward vs reverse patching comparison
│   ├── head_attribution_plot.py    # layer bar chart (max aggregation)
│   ├── logging_utils.py            # timestamped JSON logging
│   ├── agreement_dataset.py        # subject-verb agreement dataset
│   ├── agreement_corruption.py     # subject-number-flip corruption
│   ├── agreement_tracing.py        # layer sweep on agreement task
│   ├── agreement_head_scan.py      # per-head patching on agreement task
│   ├── runner.py                   # unified patch_and_score/sweep runner
│   ├── serialize.py                # ExperimentResult save/load
│   ├── final_experiment.py         # finalized Paris/London experiment
│   ├── ablation.py                 # single-head zero-ablation
│   ├── ablation_plot.py            # layer bar chart for ablation (min agg)
│   ├── candidate_circuit.py        # patching/ablation top-10 intersection
│   ├── pathway_analysis.py         # composition cosine-sim (inconclusive)
│   ├── mlp_and_heads.py            # combined head+MLP patching
│   ├── causal_graph.py             # CausalGraph node/edge additive analysis
│   ├── circuit_ranking.py          # all-subset ranking of candidate components
│   ├── refine_hypothesis.py        # 4th-component marginal-gain test
│   ├── targeted_patching.py        # position-specific + top-level-hook patching
│   ├── experiment_summary.py       # checkpoint sweep (embed -> resid_post)
│   ├── sweep_runner.py             # generalized checkpoint sweep, any task
│   ├── aggregate_effects.py        # sweep averaged over N examples
│   ├── circuit_robustness.py       # sweep per IOI template
│   ├── visualization_data.py       # side-by-side Paris/London vs IOI chart
│   └── final_attribution_graph.py  # final IOI head ranking + layer sweep
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
    ├── logs/        # gitignored
    └── results/     # gitignored

## Methodology

Full write-up of techniques (activation patching, ablation, forward vs.
reverse patching, composition analysis) and the bugs/lessons that shaped
them: [`docs/methodology.md`](docs/methodology.md).

## License

MIT - see `LICENSE`.