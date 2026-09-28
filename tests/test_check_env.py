from pcam_mets_explain.check_env import main


def test_check_env_runs(capsys):
    main()
    text = capsys.readouterr().out
    assert "PatchMets-XAI environment" in text
    assert "Environment check finished" in text
