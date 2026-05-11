from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists() and (candidate / "README.md").exists():
            return candidate
    raise FileNotFoundError("Не удалось найти корень проекта.")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Не найден обязательный файл: {path}")


def main() -> None:
    root = find_project_root()
    print(f"PROJECT_ROOT: {root}")

    required_files = [
        root / "README.md",
        root / "requirements.txt",
        root / "docs" / "model_report.md",
        root / "notebooks" / "01_save_and_parse_matches.ipynb",
        root / "notebooks" / "02_parse_rankings.ipynb",
        root / "notebooks" / "03_build_final_dataset.ipynb",
        root / "notebooks" / "04_train_final_model.ipynb",
        root / "assembled_predictor" / "predictor_model_dataset_from_parsers.xlsx",
    ]
    for path in required_files:
        require_file(path)

    for notebook_path in sorted((root / "notebooks").glob("*.ipynb")):
        with notebook_path.open(encoding="utf-8") as file:
            json.load(file)
        print(f"notebook ok: {notebook_path.relative_to(root)}")

    dataset_path = root / "assembled_predictor" / "predictor_model_dataset_from_parsers.xlsx"
    excel = pd.ExcelFile(dataset_path)
    required_sheets = {"ml_dataset", "matches_enriched", "player_matching", "coverage"}
    missing_sheets = required_sheets.difference(excel.sheet_names)
    if missing_sheets:
        raise ValueError(f"В датасете не хватает листов: {sorted(missing_sheets)}")

    coverage = pd.read_excel(dataset_path, sheet_name="coverage")
    if coverage.empty:
        raise ValueError("Лист coverage пуст.")

    print("dataset ok: assembled_predictor/predictor_model_dataset_from_parsers.xlsx")
    print("project verification passed")


if __name__ == "__main__":
    main()
