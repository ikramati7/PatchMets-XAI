from pcam_mets_explain.check_env import main


def test_check_env_runs(capsys):
    main()
    text = capsys.readouterr().out
    assert "pcam-mets-explain environment" in text
    assert "Day 0 check finished" in text
