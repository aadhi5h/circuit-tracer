import os
from src.serialize import ExperimentResult

def test_save_and_load_roundtrip():
    result = ExperimentResult(
        name="test_roundtrip",
        hook_name="resid_post",
        layer_results=[(0, 1.5), (1, 2.3)],
        head_results=[(0, 0, 0.1), (0, 1, 0.2)],
        metadata={"foo": "bar"},
    )
    path = result.save()
    assert os.path.exists(path)

    loaded = ExperimentResult.load(path)
    assert loaded.name == result.name
    assert [tuple(r) for r in loaded.layer_results] == result.layer_results
    assert [tuple(r) for r in loaded.head_results] == result.head_results
    assert loaded.metadata == result.metadata

    print("serialization round-trip test passed")
    os.remove(path)

if __name__ == "__main__":
    test_save_and_load_roundtrip()