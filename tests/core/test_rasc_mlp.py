"""Tests for RASCMLP.

Author: Ryan Vig (NetID: rvig2)

Uses small synthetic datasets, including samples shaped like the output of
ValueSetCompletion. No network access or real data.
"""

import unittest

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RASCMLP
from pyhealth.tasks.value_set_completion import CODE_SYSTEM_VOCAB, FEATURES

SCHEMA = {
    "title_embedding": "tensor",
    "candidate_embedding": "tensor",
    "code_system": "tensor",
    "score": "tensor",
}


def make_samples(n: int, dim: int = 6, seed: int = 0) -> list[dict]:
    """Pairs whose label is 1 when title and candidate vectors agree."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        title = rng.normal(size=dim).astype(np.float32)
        label = int(i % 4 == 0)  # 25% positives
        noise = rng.normal(size=dim).astype(np.float32)
        candidate = title + 0.3 * noise if label else noise
        system = np.zeros(3, dtype=np.float32)
        system[i % 3] = 1.0
        samples.append(
            {
                "patient_id": f"vs-{i // 8}",
                "title_embedding": title.tolist(),
                "candidate_embedding": candidate.tolist(),
                "code_system": system.tolist(),
                "score": [float(title @ candidate / dim)],
                "label": label,
            }
        )
    return samples


def make_dataset(samples: list[dict], pos_weight: str | None = None, **schema):
    label = ("binary", {"pos_weight": pos_weight}) if pos_weight else "binary"
    return create_sample_dataset(
        samples=samples,
        input_schema=schema or SCHEMA,
        output_schema={"label": label},
        dataset_name="rasc_mlp_test",
    )


class TestRASCMLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = make_dataset(make_samples(64))

    def batch(self, size: int = 16) -> dict:
        return next(iter(get_dataloader(self.dataset, batch_size=size)))

    def test_architecture(self):
        model = RASCMLP(self.dataset)
        self.assertEqual(model.input_dim, 6 + 6 + 3 + 1)
        linears = [m for m in model.mlp if isinstance(m, torch.nn.Linear)]
        self.assertEqual([m.out_features for m in linears], [512, 256, 64])
        self.assertEqual(model.fc.out_features, 1)
        dropouts = [m for m in model.mlp if isinstance(m, torch.nn.Dropout)]
        self.assertTrue(all(m.p == 0.3 for m in dropouts))
        self.assertEqual(
            sum(isinstance(m, torch.nn.BatchNorm1d) for m in model.mlp), 3
        )

    def test_without_batch_norm(self):
        model = RASCMLP(self.dataset, hidden_dims=(8, 4), batch_norm=False)
        self.assertFalse(any(isinstance(m, torch.nn.BatchNorm1d) for m in model.mlp))
        self.assertEqual(len(model.mlp), 2 * 3)  # Linear, ReLU, Dropout per layer
        self.assertTrue(torch.isfinite(model(**self.batch())["loss"]))

    def test_forward_and_backward(self):
        model = RASCMLP(self.dataset, hidden_dims=(8, 4))
        out = model(**self.batch(), embed=True)
        self.assertEqual(out["logit"].shape, (16, 1))
        self.assertEqual(out["y_prob"].shape, (16, 1))
        self.assertEqual(out["embed"].shape, (16, 4))
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()
        self.assertIsNotNone(model.fc.weight.grad)

    def test_without_label(self):
        model = RASCMLP(self.dataset, hidden_dims=(8,)).eval()
        batch = {k: v for k, v in self.batch().items() if k != "label"}
        out = model(**batch)
        self.assertNotIn("loss", out)
        self.assertEqual(out["y_prob"].shape, (16, 1))

    def test_feature_keys_select_and_order(self):
        model = RASCMLP(self.dataset, feature_keys=["score", "title_embedding"])
        self.assertEqual(model.feature_keys, ["score", "title_embedding"])
        self.assertEqual(model.input_dim, 7)
        self.assertEqual(model(**self.batch())["logit"].shape, (16, 1))

    def test_input_order_matters(self):
        """The first layer sees features in feature_keys order."""
        torch.manual_seed(0)
        model = RASCMLP(self.dataset, hidden_dims=(4,)).eval()
        batch = self.batch()
        with torch.no_grad():
            model.mlp[0].weight.zero_()
            model.mlp[0].weight[:, -1] = 1.0  # last input is the score
        y = model(**batch)["logit"]
        batch["title_embedding"] = torch.zeros_like(batch["title_embedding"])
        torch.testing.assert_close(model(**batch)["logit"], y)

    def test_balanced_pos_weight_loss(self):
        dataset = make_dataset(make_samples(64), pos_weight="balanced")
        model = RASCMLP(dataset, hidden_dims=(8,)).eval()
        batch = next(iter(get_dataloader(dataset, batch_size=64)))
        out = model(**batch)
        expected = F.binary_cross_entropy_with_logits(
            out["logit"], out["y_true"], pos_weight=torch.tensor([3.0])
        )
        torch.testing.assert_close(out["loss"], expected)

    def test_training_batch_of_one(self):
        model = RASCMLP(self.dataset, hidden_dims=(8,)).train()
        bn = model.mlp[1]
        before = bn.running_mean.clone()
        out = model(**self.batch(size=1))
        self.assertTrue(torch.isfinite(out["loss"]))
        torch.testing.assert_close(bn.running_mean, before)

    def test_learns_signal(self):
        torch.manual_seed(0)
        dataset = make_dataset(make_samples(256), pos_weight="balanced")
        model = RASCMLP(dataset, hidden_dims=(32, 16), dropout=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        loader = get_dataloader(dataset, batch_size=64, shuffle=True)
        for _ in range(15):
            for batch in loader:
                optimizer.zero_grad()
                model(**batch)["loss"].backward()
                optimizer.step()
        model.eval()
        batch = next(iter(get_dataloader(dataset, batch_size=256)))
        with torch.no_grad():
            out = model(**batch)
        auroc = roc_auc_score(out["y_true"].ravel(), out["y_prob"].ravel())
        self.assertGreater(auroc, 0.9)

    def test_invalid_arguments(self):
        with self.assertRaisesRegex(ValueError, "Unknown feature_keys"):
            RASCMLP(self.dataset, feature_keys=["missing"])
        with self.assertRaisesRegex(ValueError, "at least one"):
            RASCMLP(self.dataset, feature_keys=[])
        with self.assertRaisesRegex(ValueError, "hidden_dims"):
            RASCMLP(self.dataset, hidden_dims=(8, 0))
        with self.assertRaisesRegex(ValueError, "dropout"):
            RASCMLP(self.dataset, dropout=1.0)

    def test_rejects_non_vector_inputs(self):
        codes = [
            {"patient_id": "p", "codes": ["a", "b"], "x": [1.0], "label": i % 2}
            for i in range(4)
        ]
        dataset = make_dataset(codes, codes="sequence", x="tensor")
        with self.assertRaisesRegex(ValueError, "'codes' must be a 'tensor'"):
            RASCMLP(dataset)
        self.assertEqual(RASCMLP(dataset, feature_keys=["x"]).input_dim, 1)

        matrices = [
            {"patient_id": "p", "m": [[1.0, 2.0], [3.0, 4.0]], "label": i % 2}
            for i in range(4)
        ]
        with self.assertRaisesRegex(ValueError, "one vector per sample"):
            RASCMLP(make_dataset(matrices, m="tensor"))


class TestRASCMLPWithValueSetCompletionSamples(unittest.TestCase):
    """RASCMLP on samples shaped like ValueSetCompletion's output."""

    def test_trains_on_task_shaped_samples(self):
        dim = 16  # stands in for SapBERT's 768
        rng = np.random.default_rng(0)
        samples = []
        for i in range(8):
            one_hot = np.zeros(len(CODE_SYSTEM_VOCAB), dtype=np.float32)
            one_hot[i % len(CODE_SYSTEM_VOCAB)] = 1.0
            samples.append(
                {
                    "patient_id": f"2.16.840.1.{i // 4}",
                    "code": f"C{i}",
                    "system": "http://snomed.info/sct",
                    "n_true_codes": 3,
                    "title_embedding": rng.normal(size=dim).astype(np.float32),
                    "candidate_embedding": rng.normal(size=dim).astype(np.float32),
                    "code_system": one_hot,
                    "score": np.array([rng.random()], dtype=np.float32),
                    "label": i % 2,
                }
            )
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={name: "tensor" for name in FEATURES},
            output_schema={"label": ("binary", {"pos_weight": "balanced"})},
            dataset_name="rasc_task_shaped",
        )
        model = RASCMLP(dataset, hidden_dims=(8,))
        self.assertEqual(model.input_dim, 2 * dim + len(CODE_SYSTEM_VOCAB) + 1)
        out = model(**next(iter(get_dataloader(dataset, batch_size=8))))
        self.assertEqual(out["y_prob"].shape, (8, 1))
        out["loss"].backward()


if __name__ == "__main__":
    unittest.main()
