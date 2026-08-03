"""Entry point for running the railway simulation."""

import runpy
from pathlib import Path


def main() -> None:
    simulation_file = (
        Path(__file__).parent
        / "src"
        / "revised_code_time_table.py"
    )

    if not simulation_file.exists():
        raise FileNotFoundError(
            f"Simulation file was not found: {simulation_file}"
        )

    runpy.run_path(
        str(simulation_file),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
