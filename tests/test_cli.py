import importlib

def test_cli_loads():
    cli = importlib.import_module("ai_doc_composer.cli")
    assert hasattr(cli, "app"), "Typer app not found"