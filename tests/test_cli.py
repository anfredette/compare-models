from __future__ import annotations

import pytest
from click.testing import CliRunner

from compare_models.cli import main


@pytest.mark.unit
class TestCLI:
    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--models" in result.output

    def test_no_models_error(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code != 0
