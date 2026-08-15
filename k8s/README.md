# Kubernetes examples

These manifests are generic, public examples for the ME344 experiment. Substitute the
`${...}` variables with environment-specific values before applying them. Keep credentials
in Workload Identity or a secrets manager, not in these files.

Supply every custom image variable as an immutable digest reference such as
`registry.example/me344ricechem:gpu@sha256:<64 hex characters>`. The checked-in vLLM
serving images and both Dockerfile base images are already digest-pinned.

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

## Endpoint exposure

The Python endpoints bind to `127.0.0.1` by default. The Kubernetes examples explicitly
bind to `0.0.0.0` so kubelet probes and `kubectl port-forward` can reach the pod, but the
checked-in manifests intentionally define no `Service` or `Ingress`. Keep them on a private
cluster network and access them through a local port-forward, for example:

```bash
kubectl port-forward deployment/ricechem-serve-gpu-4b 8000:8000
```

Do not expose these unauthenticated inference endpoints with a public `LoadBalancer`.
If an internal service is required, pair it with a restrictive `NetworkPolicy` and an
authenticated gateway. The 27B training/serve job also requires a shutdown token from the
Kubernetes Secret named by `${SHUTDOWN_SECRET_NAME}`; only the process-control route uses
that token. Model storage is mounted read-only in serving pods, while TPU XLA compilation
uses a separate writable `emptyDir` cache.
