# Guardrails detector API contracts

Outcome **A: provider**. The public contract in scope is
`POST /api/v1/text/contents` for the Hugging Face detector service. Its
generated OpenAPI document is registered as `huggingface-detector-v1` with
`matchPath: '^/api/v1/text/contents$'`.

## Consumer evidence

- This repository's [README](../README.md) identifies the FMS Guardrails
  Orchestrator as the consumer of the detector microservices.
- The orchestrator's [detector client](https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/b60108e9bd83faac715c71e10dc9ff0fa1fc1610/src/clients/detector.rs#L279)
  sends `ContentAnalysisRequest` to `/api/v1/text/contents` and deserializes
  `Vec<Vec<ContentAnalysisResponse>>`. The endpoint constant is declared at
  [line 47](https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/b60108e9bd83faac715c71e10dc9ff0fa1fc1610/src/clients/detector.rs#L47).
- The actual provider route is in `detectors/huggingface/app.py`; its
  request and response models are in `detectors/common/scheme.py`.
- Discovery reports one standalone OpenAPI document and zero CRD manifests.
  Deployment manifests instantiate KServe resources; this repository does not
  own their CRDs. No documents are excluded.

## Generate and verify

Use Python 3.12 in a virtual environment:

```sh
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  -r scripts/openapi-requirements.txt './detectors[huggingface]'
python scripts/generate_openapi.py
python scripts/generate_openapi.py --check
```

The generator imports the unchanged Hugging Face application and uses its real
FastAPI route and Pydantic models, restricted to the text-contents path. Its
dependencies, including CPU-only PyTorch, must be installed. The application
lifespan is not entered, so generation does not load or download model weights.
Commit the regenerated document with every API change.

`scripts/openapi-requirements.txt` pins the schema-generation environment from
`detectors/pyproject.toml`, including the `huggingface` extra. Refresh it after
changing dependencies using the
`uv pip compile` command recorded at the top of that file, then regenerate.
CI installs both the pins and the project so incompatible dependency pins fail.
Keep index options out of the generated requirements file: pip's `--index-url`
directive there resets the index list and discards the CPU wheel index supplied
by the installation command. Do not add `--emit-index-url` when regenerating.

The central `openapi.static` adapter reads the committed artifact. The
separate `generated-openapi` job enforces freshness before compatibility checks;
the central runner does not generate FastAPI schemas. Every reusable contract
job explicitly uses `mode: enforce` and `@latest`.

## Baseline and local results

Verified on 2026-11-17 against `opendatahub-io/guardrails-detectors`: the newest
GA tag is `v0.3.0+rhaiv.4`, resolving to
`3c39a53d7fe278242bdb9d6e22b3e24c4bddb177`. This follows the ticket's GA-tag-first
rule. The SHA and source ref are recorded beside the workflow inputs.

Local validation used central release `v2.2.0` (`@latest` resolved to
`d63fe27957117fd99161a693c7fdef9e20efb9d9`), oasdiff `1.33.0`, and yq `4.45.1`:

| Contract | Artifact | Result |
| --- | --- | --- |
| huggingface-detector-v1 | api/openapi/huggingface.json | new |

Coverage: `Result: complete`. No baseline overrides or exclusions.

The API already existed, but no OpenAPI document was committed at the GA
baseline. Consequently `new` means **no historical schema comparison was
possible**, not that compatibility with the released implementation was proven.
Until an owner other than the author approves a baseline containing this
document, the checks validate the candidate but cannot detect changes relative
to a released schema. Keep the selected baseline unchanged during onboarding.

To reproduce, set `ACT` to a checkout of the pinned central release, `TOOLS` to
the directory containing its pinned binaries, and `REPORTS` to a fresh directory:

```sh
python "$ACT/cmd/contract-tests/cli.py" check --repo . \
  --contract-set .rhoai/contracts.yaml --contract-id huggingface-detector-v1 \
  --baseline-ref 3c39a53d7fe278242bdb9d6e22b3e24c4bddb177 \
  --oasdiff "$TOOLS/oasdiff" --yq "$TOOLS/yq" \
  --output-dir "$REPORTS/huggingface-detector-v1" --enforce
python "$ACT/cmd/contract-tests/cli.py" coverage --repo . \
  --contract-set .rhoai/contracts.yaml \
  --baseline-ref 3c39a53d7fe278242bdb9d6e22b3e24c4bddb177 \
  --yq "$TOOLS/yq" --yq-version 4.45.1
```

## Ownership and registration

`@opendatahub-io/trustyai` owns the contract policy, workflow, and generated
artifacts. Repository branch protection must require code-owner review for that
rule to block merges. Preserve this onboarding configuration and its generator
when syncing from upstream.

Working-group registration pointer: repository
`opendatahub-io/guardrails-detectors`, descriptor `.rhoai/contracts.yaml`, and
the eventual merged revision containing these files. Registration and the
ticket comment must link that revision; they are not complete merely because
this descriptor exists.
