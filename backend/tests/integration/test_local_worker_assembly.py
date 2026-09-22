"""Phase D2 contract: the launcher worker is the real local assembly.

The manifest loader must re-validate authorization/review/duration from
persisted SQLite segments, the evaluation stage must fail closed with a fixed
message while local evaluation weights are absent, the reference resolver must
fail closed until a profile publishes its local reference index, and the
assembled worker must carry the real adapters instead of placeholders.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.app.api.routes.datasets import DatasetService
from backend.app.services.gpt_sovits import GPTSoVITSAdapter
from backend.app.services.gpt_sovits_dataset import GPTSoVITSDatasetPreparer
from backend.app.services.disentanglement import DisentanglerTrainer
from backend.app.services.local_worker import (
    SegmentManifestLoader,
    assemble_local_worker,
)
from backend.app.services.storage import LocalStorage
from backend.app.services.synthesis_pipeline import SynthesisPipeline
from backend.app.services.training_errors import TrainingPipelineError
from backend.app.workers.gpu_worker import GPUWorker


ROOT = Path(__file__).resolve().parents[3]


def _service(tmp_path: Path) -> DatasetService:
    return DatasetService(storage=LocalStorage(tmp_path / "data"))


def test_manifest_loader_requires_authorization(tmp_path):
    service = _service(tmp_path)
    service.register_dataset(
        "d1", effective_seconds=600, owner_user_id=None, authorization_confirmed=False
    )
    loader = SegmentManifestLoader(service)

    with pytest.raises(TrainingPipelineError, match="授权"):
        loader.load("d1")


def test_manifest_loader_requires_reviewed_non_empty_segments(tmp_path):
    service = _service(tmp_path)
    # authorized and duration seeded, but no reviewed segment text rows exist
    # when the seeded segment is removed from the manifest build.
    service.register_dataset(
        "d2", effective_seconds=600, owner_user_id=None, authorization_confirmed=True
    )
    with service.engine.begin() as connection:
        connection.exec_driver_sql("DELETE FROM dataset_segments")
    loader = SegmentManifestLoader(service)

    with pytest.raises(TrainingPipelineError, match="审核"):
        loader.load("d2")


def test_manifest_loader_revalidates_duration_window(tmp_path):
    service = _service(tmp_path)
    service.register_dataset(
        "d3", effective_seconds=479, owner_user_id=None, authorization_confirmed=True
    )
    loader = SegmentManifestLoader(service)

    with pytest.raises(TrainingPipelineError, match="480"):
        loader.load("d3")


def test_manifest_loader_returns_reviewed_manifest(tmp_path):
    service = _service(tmp_path)
    service.register_dataset(
        "d4", effective_seconds=600, owner_user_id=None, authorization_confirmed=True
    )
    segment_file = tmp_path / "data" / "datasets" / "d4" / "segments" / "seg_0001.wav"
    segment_file.parent.mkdir(parents=True, exist_ok=True)
    segment_file.write_bytes(b"RIFF")
    loader = SegmentManifestLoader(service)

    manifest = loader.load("d4")

    assert manifest.dataset_id == "d4"
    assert len(manifest.rows) == 1
    assert manifest.rows[0].text == "种子审核分段文本。"
    assert manifest.effective_seconds == pytest.approx(600.0)


def test_evaluation_stage_fails_closed_without_local_models(tmp_path):
    from backend.app.services.local_worker import LocalTrainingAdapters

    adapters = LocalTrainingAdapters(
        gpt_sovits=GPTSoVITSAdapter(vendor_dir=tmp_path / "vendor"),
        campplus_dir=tmp_path / "models" / "campplus",
        emotion_dir=tmp_path / "models" / "emotion2vec",
        models_root=tmp_path / "models",
        storage_root=tmp_path / "data",
    )
    manifest = SimpleNamespace(rows=[], effective_seconds=600.0, dataset_id="d1")

    with pytest.raises(TrainingPipelineError, match="评测模型"):
        adapters.evaluate(manifest, tmp_path)


def test_reference_resolver_fails_closed_without_local_index(tmp_path):
    from backend.app.services.local_worker import LocalProfileReferenceResolver

    resolver = LocalProfileReferenceResolver()
    profile = SimpleNamespace(
        id="p1",
        public_weight_dir=str(tmp_path / "data" / "profiles" / "p1" / "public"),
    )

    with pytest.raises(TrainingPipelineError, match="参考"):
        resolver.resolve(
            profile,
            SimpleNamespace(mode="manual", label=None, strength=0.5, reference_audio_id=None),
        )


def test_assemble_local_worker_wires_real_components(tmp_path):
    (tmp_path / "data").mkdir(parents=True)
    worker = assemble_local_worker(root=tmp_path)

    assert isinstance(worker, GPUWorker)
    assert isinstance(worker.dataset_preparer, GPTSoVITSDatasetPreparer)
    assert isinstance(worker.gpt_sovits_adapter, GPTSoVITSAdapter)
    assert isinstance(worker.disentangler_trainer, DisentanglerTrainer)
    assert isinstance(worker.synthesis_pipeline, SynthesisPipeline)
    assert worker.evaluation_runner is not None
    assert not hasattr(worker.adapters, "__dict__") or type(worker.adapters).__name__ != (
        "UnconfiguredAdapters"
    )


def test_assemble_local_worker_uses_explicit_storage_root(tmp_path):
    storage_root = tmp_path / "isolated-data"
    worker = assemble_local_worker(root=tmp_path, storage_root=storage_root)

    assert worker.synthesis_pipeline is not None
    assert worker.synthesis_pipeline.storage.root == storage_root.resolve()
    assert worker.profiles_root == (storage_root / "profiles").resolve()
    assert worker.features_root == (storage_root / "features").resolve()


def test_assemble_local_worker_provides_callable_reviewed_manifest_loader(tmp_path):
    """The assembled worker must pass a callable into GPUWorker, not the loader object."""
    service = _service(tmp_path)
    service.register_dataset(
        "assembled-dataset",
        effective_seconds=600,
        owner_user_id=None,
        authorization_confirmed=True,
    )
    segment_file = (
        tmp_path / "data" / "datasets" / "assembled-dataset" / "segments" / "seg_0001.wav"
    )
    segment_file.parent.mkdir(parents=True, exist_ok=True)
    segment_file.write_bytes(b"RIFF")

    worker = assemble_local_worker(root=tmp_path)

    assert callable(worker.manifest_loader)
    manifest = worker.manifest_loader("assembled-dataset")
    assert manifest.dataset_id == "assembled-dataset"
