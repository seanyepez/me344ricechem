# Kubernetes examples

These manifests are generic, public examples for the ME344 experiment. Substitute the
`${...}` variables with environment-specific values before applying them. Keep credentials
in Workload Identity or a secrets manager, not in these files.

The examples enforce the experiment's data boundary:

- only train and validation chunks may be mounted into training jobs;
- canonical test prompts remain on the authorized client and stream to a private endpoint;
- adapters/checkpoints go to an access-controlled model bucket;
- raw predictions and RiceChem rows are never published.

Example rendering:

```bash
envsubst < k8s/train-gpu-4b.yaml > /tmp/train-gpu-4b.rendered.yaml
kubectl apply --dry-run=server -f /tmp/train-gpu-4b.rendered.yaml
```

The public repository does not include project IDs, service accounts, buckets, or image
registry paths.
