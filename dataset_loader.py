import pandas as pd
from pathlib import Path



class DatasetValidationError(Exception):
    """Raised when dataset validation fails."""
    pass


class DatasetValidator:
    REQUIRED_COLUMNS = ["image_path", "mask_path"]
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    @staticmethod
    def validate_dataframe(df: pd.DataFrame):
        if df.empty:
            raise DatasetValidationError("Dataset is empty.")

        for col in DatasetValidator.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise DatasetValidationError(
                    f"Missing required column: {col}"
                )

    @staticmethod
    def validate_paths(df: pd.DataFrame):
        for _, row in df.iterrows():
            if not Path(row["image_path"]).exists():
                raise DatasetValidationError(
                    f"Missing image: {row['image_path']}"
                )

            if not Path(row["mask_path"]).exists():
                raise DatasetValidationError(
                    f"Missing mask: {row['mask_path']}"
                )
    @staticmethod
    def validate_extensions(df: pd.DataFrame):
        for _, row in df.iterrows():
            image_ext = Path(row["image_path"]).suffix.lower()
            mask_ext = Path(row["mask_path"]).suffix.lower()

            if image_ext not in DatasetValidator.SUPPORTED_EXTENSIONS:
                raise DatasetValidationError(
                    f"Unsupported image format: {image_ext}"
                )

            if mask_ext not in DatasetValidator.SUPPORTED_EXTENSIONS:
                raise DatasetValidationError(
                    f"Unsupported mask format: {mask_ext}"
                )
            
class BaseDatasetLoader:
    def load(self, dataset_path="."):
        raise NotImplementedError
               

class TCGALoader(BaseDatasetLoader):
    def load(self, dataset_path="."):
        csv_path = Path(dataset_path) / "data_mask.csv"

        if not csv_path.exists():
            raise DatasetValidationError(
                f"Missing file: {csv_path}"
            )
        df = pd.read_csv(csv_path)

        DatasetValidator.validate_dataframe(df)
        DatasetValidator.validate_paths(df)
        DatasetValidator.validate_extensions(df)

        df["dataset_source"] = "TCGA"
        return df
    
class BraTSLoader(BaseDatasetLoader):
    def load(self, dataset_path="."):
        dataset_path = Path(dataset_path)

        images_dir = dataset_path / "images"
        masks_dir = dataset_path / "masks"

        if not images_dir.exists():
            raise DatasetValidationError(
                f"Missing images directory: {images_dir}"
            )

        if not masks_dir.exists():
            raise DatasetValidationError(
                f"Missing masks directory: {masks_dir}"
            )

        image_files = sorted(images_dir.glob("*"))
        mask_files = sorted(masks_dir.glob("*"))

        if not image_files:
            raise DatasetValidationError(
                "No MRI images found."
            )

        if len(image_files) != len(mask_files):
            raise DatasetValidationError(
                "Image and mask counts do not match."
            )

        df = pd.DataFrame({
            "image_path": [str(f) for f in image_files],
            "mask_path": [str(f) for f in mask_files],
        })

        DatasetValidator.validate_dataframe(df)
        DatasetValidator.validate_paths(df)
        DatasetValidator.validate_extensions(df)

        df["dataset_source"] = "BraTS"

        return df

class MultiDatasetLoader:
    LOADERS = {
        "tcga": TCGALoader,
        "brats": BraTSLoader,
    }
    def load_dataset(self, dataset_type, dataset_path="."):
        """
        Load a dataset using a unified interface.

        Parameters:
            dataset_type (str): Dataset identifier (tcga, brats)
            dataset_path (str): Path to dataset directory

        Returns:
            pandas.DataFrame
        """
        dataset_type = dataset_type.lower()

        if dataset_type not in self.LOADERS:
            raise ValueError(
                f"Unsupported dataset type: {dataset_type}"
            )

        loader = self.LOADERS[dataset_type]()

        return loader.load(dataset_path)
    
if __name__ == "__main__":
    loader = MultiDatasetLoader()
    print("Dataset loader module ready.")
    