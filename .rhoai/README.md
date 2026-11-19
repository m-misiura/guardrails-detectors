# Guardrails detector API contracts

Outcome **A: provider**. The public contract is `POST /api/v1/text/contents`,
registered as `huggingface-detector-v1` with
`matchPath: '^/api/v1/text/contents$'`.

The generated OpenAPI artifact and freshness workflow are maintained upstream.
This downstream onboarding intentionally does not duplicate them.

Baseline: `3c39a53d7fe278242bdb9d6e22b3e24c4bddb177`
(`v0.3.0+rhaiv.4`). Coverage is complete and the contract is reported as new
because the baseline did not include an OpenAPI document.

`@opendatahub-io/trustyai` owns the downstream contract policy and workflow.
