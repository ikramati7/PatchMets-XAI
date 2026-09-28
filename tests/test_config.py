from pcam_mets_explain.config import CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES, ROOT


def test_image_size_matches_pcam():
    assert IMAGE_SIZE == 96
    assert NUM_CLASSES == 2
    assert CLASS_NAMES == ("no_metastasis", "metastasis")
    assert ROOT.name == "pcam-mets-explain"
