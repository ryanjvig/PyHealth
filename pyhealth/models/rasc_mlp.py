"""RASC candidate classifier: one MLP over concatenated dense features.

Author: Ryan Vig (NetID: rvig2)
Paper: Retrieve, Then Classify: Corpus-Grounded Automation of Clinical Value
    Set Authoring (Mukherjee et al., 2026)
Paper link: https://arxiv.org/abs/2604.14616
Description: The paper's MLP baseline for value set completion. It
    concatenates the title embedding, candidate display-name embedding,
    code-system one-hot and retrieval score (1,545 dims with SapBERT) and
    scores the pair with a 512-256-64 MLP with BatchNorm, ReLU and dropout.
    PyHealth's :class:`~pyhealth.models.MLP` instead runs one tower per input
    with no BatchNorm or dropout. Written from the paper's description; no
    code is taken from the authors' repository.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.processors import TensorProcessor

from .base_model import BaseModel


class _BatchNorm1d(nn.BatchNorm1d):
    """BatchNorm that uses running statistics for a training batch of one.

    ``nn.BatchNorm1d`` raises on a training batch of size 1, which happens
    whenever the training set size is one more than a multiple of the batch
    size. Such a batch is normalized with the running statistics and leaves
    them unchanged.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.size(0) == 1:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )
        return super().forward(x)


class RASCMLP(BaseModel):
    """MLP over the concatenation of dense per-sample feature vectors.

    The model of the RASC paper's MLP baseline (Mukherjee et al., 2026). It
    concatenates every input in ``feature_keys`` into one vector and applies
    ``Linear -> BatchNorm -> ReLU -> Dropout`` per hidden layer, then a
    linear output layer. With :class:`~pyhealth.tasks.ValueSetCompletion`
    and SapBERT the input is the paper's 1,545-dim vector: title embedding
    (768), candidate embedding (768), code-system one-hot (8) and retrieval
    score (1).

    The loss and output activation come from the dataset's label processor.
    The task's ``pos_weight="balanced"`` gives the paper's class-balanced
    BCE. The paper trains with AdamW (lr 3e-4), batch size 2,048 and early
    stopping with patience 5; pass those to
    :class:`~pyhealth.trainer.Trainer`.

    Args:
        dataset: The dataset to train the model. Every input in
            ``feature_keys`` must be a ``"tensor"`` input with one value
            vector per sample.
        feature_keys: Inputs to concatenate, in order. Defaults to all
            inputs of ``dataset``.
        hidden_dims: Sizes of the hidden layers. Default ``(512, 256, 64)``.
        dropout: Dropout probability after each hidden layer. Default 0.3.
        batch_norm: Whether each hidden layer has BatchNorm. Default True,
            as in the paper; False is an ablation.

    Raises:
        ValueError: If ``dataset`` has more than one label, ``feature_keys``
            is empty or names an unknown input, an input is not a 1-D
            ``"tensor"`` feature, ``hidden_dims`` has a non-positive size,
            or ``dropout`` is outside ``[0, 1)``.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import RASCMLP
        >>> samples = [
        ...     {
        ...         "patient_id": f"vs-{i}",
        ...         "title_embedding": [0.1 * i, 0.2, 0.3, 0.4],
        ...         "candidate_embedding": [0.4, 0.3, 0.2, 0.1 * i],
        ...         "score": [0.5 + 0.1 * i],
        ...         "label": i % 2,
        ...     }
        ...     for i in range(4)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "title_embedding": "tensor",
        ...         "candidate_embedding": "tensor",
        ...         "score": "tensor",
        ...     },
        ...     output_schema={"label": "binary"},
        ...     dataset_name="rasc_doctest",
        ... )
        >>> model = RASCMLP(dataset, hidden_dims=(16, 8))
        >>> model.input_dim
        9
        >>> batch = next(iter(get_dataloader(dataset, batch_size=4)))
        >>> out = model(**batch)
        >>> sorted(out)
        ['logit', 'loss', 'y_prob', 'y_true']
        >>> out["y_prob"].shape
        torch.Size([4, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_keys: Sequence[str] | None = None,
        hidden_dims: Sequence[int] = (512, 256, 64),
        dropout: float = 0.3,
        batch_norm: bool = True,
    ) -> None:
        super().__init__(dataset)
        if len(self.label_keys) != 1:
            raise ValueError(
                f"RASCMLP needs exactly one label, got {self.label_keys}."
            )
        self.label_key = self.label_keys[0]
        if feature_keys is not None:
            unknown = [key for key in feature_keys if key not in self.feature_keys]
            if unknown:
                raise ValueError(f"Unknown feature_keys {unknown}.")
            self.feature_keys = list(feature_keys)
        if not self.feature_keys:
            raise ValueError("RASCMLP needs at least one input feature.")
        if any(size < 1 for size in hidden_dims):
            raise ValueError(f"hidden_dims must be positive, got {hidden_dims}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.feature_dims = {
            key: self._vector_size(key) for key in self.feature_keys
        }
        self.input_dim = sum(self.feature_dims.values())
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        self.batch_norm = batch_norm

        layers: list[nn.Module] = []
        width = self.input_dim
        for size in self.hidden_dims:
            layers.append(nn.Linear(width, size))
            if batch_norm:
                layers.append(_BatchNorm1d(size))
            layers += [nn.ReLU(), nn.Dropout(dropout)]
            width = size
        self.mlp = nn.Sequential(*layers)
        self.fc = nn.Linear(width, self.label_processor.output_size())

    def _vector_size(self, key: str) -> int:
        """Returns the width of a 1-D ``"tensor"`` input.

        Args:
            key: Input name.

        Returns:
            The number of values per sample.

        Raises:
            ValueError: If the input is not a fitted 1-D tensor feature.
        """
        processor = self.dataset.input_processors[key]
        shape = getattr(processor, "_shape", None)
        if not isinstance(processor, TensorProcessor) or shape is None:
            raise ValueError(
                f"Input {key!r} must be a 'tensor' feature with a fixed shape."
            )
        if len(shape) != 1:
            raise ValueError(
                f"Input {key!r} must be one vector per sample, got shape {shape}."
            )
        return processor.size()

    def forward(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Scores a batch.

        Args:
            **kwargs: A batch from :func:`~pyhealth.datasets.get_dataloader`.
                Must contain every input in ``feature_keys``; the label is
                optional. Pass ``embed=True`` to also return the last hidden
                layer.

        Returns:
            ``logit`` and ``y_prob`` of shape ``(batch, output_size)``, plus
            ``loss`` and ``y_true`` when the batch has the label, and
            ``embed`` when requested.
        """
        x = torch.cat(
            [
                torch.as_tensor(kwargs[key], dtype=torch.float32)
                .to(self.device)
                .reshape(-1, self.feature_dims[key])
                for key in self.feature_keys
            ],
            dim=1,
        )
        hidden = self.mlp(x)
        logit = self.fc(hidden)
        results = {"logit": logit, "y_prob": self.label_processor.activation(logit)}
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            results["loss"] = self.label_processor.loss(logit, y_true)
            results["y_true"] = y_true
        if kwargs.get("embed", False):
            results["embed"] = hidden
        return results
