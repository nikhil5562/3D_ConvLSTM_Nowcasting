import importlib


pipeline = importlib.import_module("3d_data_scripts.run_l2b_to_ml_pipeline")


def test_pipeline_rejects_missing_input(tmp_path, capsys):
    code = pipeline.main(["--input", str(tmp_path / "missing")])
    assert code == 2
    assert "does not exist" in capsys.readouterr().out


def test_pipeline_stops_before_gridding_when_no_sequence(tmp_path, monkeypatch):
    candidate = tmp_path / "RCTLS_09JUN2026_000946_L2B_STD.nc"
    candidate.touch()
    monkeypatch.setattr(pipeline, "discover_l2b_candidates", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(
        pipeline,
        "diagnose_sequence_capacity",
        lambda *args, **kwargs: {
            "frame_count": 1,
            "run_lengths": [1],
            "longest_run": 1,
            "sequence_count": 0,
        },
    )
    monkeypatch.setattr(
        pipeline,
        "create_gridded_cubes",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not grid")),
    )

    assert pipeline.main(["--input", str(tmp_path)]) == 3


def test_pipeline_qa_override_allows_single_scan(tmp_path, monkeypatch):
    candidate = tmp_path / "RCTLS_09JUN2026_000946_L2B_STD.nc"
    candidate.touch()
    cube = tmp_path / "cube.nc"
    tensor = tmp_path / "tensor.npy"
    monkeypatch.setattr(pipeline, "discover_l2b_candidates", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(
        pipeline,
        "diagnose_sequence_capacity",
        lambda *args, **kwargs: {
            "frame_count": 1,
            "run_lengths": [1],
            "longest_run": 1,
            "sequence_count": 0,
        },
    )
    monkeypatch.setattr(pipeline, "create_gridded_cubes", lambda **kwargs: [cube])
    monkeypatch.setattr(pipeline, "tensorize_cube", lambda *args, **kwargs: tensor)

    code = pipeline.main(
        ["--input", str(tmp_path), "--allow-insufficient-sequence"]
    )
    assert code == 0
