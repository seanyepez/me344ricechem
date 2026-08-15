import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EndpointSecurityTest(unittest.TestCase):
    def test_python_endpoints_default_to_loopback(self):
        for relative in ("src/serve_transformers.py", "src/train_gpu_27b.py"):
            source = (ROOT / relative).read_text()
            ast.parse(source)
            self.assertIn('os.environ.get("BIND_HOST", "127.0.0.1")', source)
            self.assertNotIn('ThreadingHTTPServer(("0.0.0.0"', source)

    def test_external_training_bind_requires_shutdown_token(self):
        source = (ROOT / "src" / "train_gpu_27b.py").read_text()
        self.assertIn("and not SHUTDOWN_TOKEN", source)
        self.assertIn("hmac.compare_digest", source)

        manifest = (ROOT / "k8s" / "train-gpu-27b.yaml").read_text()
        self.assertIn('name: BIND_HOST, value: "0.0.0.0"', manifest)
        self.assertIn("name: SHUTDOWN_TOKEN", manifest)
        self.assertIn("secretKeyRef:", manifest)

    def test_cpu_kubernetes_bind_is_an_explicit_override(self):
        manifest = (ROOT / "k8s" / "serve-cpu-4b.yaml").read_text()
        self.assertIn('name: BIND_HOST, value: "0.0.0.0"', manifest)

    def test_serving_model_mounts_are_read_only(self):
        for name in ("serve-cpu-4b.yaml", "serve-gpu-4b.yaml", "serve-tpu-4b.yaml"):
            manifest = (ROOT / "k8s" / name).read_text()
            storage_mounts = [
                line.strip()
                for line in manifest.splitlines()
                if "name: storage, mountPath: /storage" in line
            ]
            self.assertEqual(len(storage_mounts), 1)
            self.assertIn("readOnly: true", storage_mounts[0])

    def test_tpu_cache_is_not_written_to_model_storage(self):
        manifest = (ROOT / "k8s" / "serve-tpu-4b.yaml").read_text()
        self.assertIn('VLLM_XLA_CACHE_PATH, value: "/xla-cache"', manifest)
        self.assertIn("name: xla-cache, mountPath: /xla-cache", manifest)


if __name__ == "__main__":
    unittest.main()
