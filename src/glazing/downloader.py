"""Dataset downloaders for linguistic resources.

This module provides automatic downloading capabilities for FrameNet, PropBank,
VerbNet, and WordNet datasets. Each downloader handles version tracking,
progress indication, and archive extraction.

Classes
-------
BaseDownloader
    Abstract base class for dataset downloaders.
VerbNetDownloader
    Downloads VerbNet from GitHub with commit hash versioning.
PropBankDownloader
    Downloads PropBank from GitHub with commit hash versioning.
WordNetDownloader
    Downloads WordNet 3.1 from Princeton University.
FrameNetDownloader
    Provides instructions for manual FrameNet download (license required).

Functions
---------
download_dataset
    Download a specific dataset by name.
download_all
    Download all available datasets.
get_downloader
    Get downloader instance for a dataset.

Examples
--------
>>> from glazing.downloader import download_dataset
>>> path = download_dataset("verbnet", Path("data/raw"))
>>> print(f"VerbNet downloaded to: {path}")

>>> from glazing.downloader import VerbNetDownloader
>>> downloader = VerbNetDownloader()
>>> path = downloader.download(Path("data/raw"))
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast

import requests
from tqdm import tqdm

from glazing.types import DatasetType


class DownloadError(Exception):
    """Raised when a download operation fails."""


class ExtractionError(Exception):
    """Raised when archive extraction fails."""


class BaseDownloader(ABC):
    """Abstract base class for dataset downloaders.

    Provides common functionality for downloading and extracting datasets
    with progress tracking and error handling.

    Attributes
    ----------
    dataset_name : str
        Human-readable name of the dataset.
    version : str
        Version string or commit hash for the dataset.

    Methods
    -------
    download(output_dir)
        Download the dataset to the specified directory.
    """

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Name of the dataset.

        Returns
        -------
        str
            Human-readable dataset name.
        """

    @property
    @abstractmethod
    def version(self) -> str:
        """Version or commit hash.

        Returns
        -------
        str
            Version identifier for reproducible downloads.
        """

    @abstractmethod
    def download(self, output_dir: Path) -> Path:
        """Download dataset to output directory.

        Parameters
        ----------
        output_dir : Path
            Directory to download the dataset to.

        Returns
        -------
        Path
            Path to the downloaded and extracted dataset.

        Raises
        ------
        DownloadError
            If download fails.
        ExtractionError
            If archive extraction fails.
        """

    def _download_file(
        self,
        url: str,
        output_path: Path,
        expected_size: int | None = None,
        chunk_size: int = 8192,
    ) -> None:
        """Download file with progress bar.

        Parameters
        ----------
        url : str
            URL to download from.
        output_path : Path
            Local path to save the file.
        expected_size : int | None, default=None
            Expected file size in bytes for progress tracking.
        chunk_size : int, default=8192
            Size of chunks to download at a time.

        Raises
        ------
        DownloadError
            If download fails or file size doesn't match expected.
        """
        try:
            # Create parent directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            if expected_size and total_size != expected_size:
                msg = f"Expected {expected_size} bytes, got {total_size}"
                raise DownloadError(msg)

            with (
                output_path.open("wb") as f,
                tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {output_path.name}",
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
                        pbar.update(len(chunk))

        except requests.RequestException as e:
            msg = f"Failed to download {url}: {e}"
            raise DownloadError(msg) from e
        except OSError as e:
            msg = f"Failed to write file {output_path}: {e}"
            raise DownloadError(msg) from e

    def _extract_archive(self, archive_path: Path, output_dir: Path) -> Path:
        """Extract archive and return extracted directory path.

        Parameters
        ----------
        archive_path : Path
            Path to the archive file.
        output_dir : Path
            Directory to extract to.

        Returns
        -------
        Path
            Path to the extracted directory.

        Raises
        ------
        ExtractionError
            If extraction fails or archive format is unsupported.
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                if archive_path.suffix.lower() == ".zip":
                    with zipfile.ZipFile(archive_path, "r") as zip_ref:
                        zip_ref.extractall(temp_path)  # noqa: S202
                elif archive_path.suffix.lower() in {".tar", ".gz"}:
                    if archive_path.name.endswith(".tar.gz"):
                        with tarfile.open(str(archive_path), "r:gz") as tar_ref:
                            tar_ref.extractall(temp_path, filter="data")
                    else:
                        with tarfile.open(str(archive_path), "r") as tar_ref:
                            tar_ref.extractall(temp_path, filter="data")
                else:
                    msg = f"Unsupported archive format: {archive_path.suffix}"
                    raise ExtractionError(msg)

                # Find the extracted directory (should be only one top-level dir)
                extracted_items = list(temp_path.iterdir())
                if len(extracted_items) != 1 or not extracted_items[0].is_dir():
                    msg = f"Expected single directory in archive, found: {extracted_items}"
                    raise ExtractionError(msg)

                source_dir = extracted_items[0]
                target_dir = output_dir / source_dir.name

                # Remove existing directory if it exists
                if target_dir.exists():
                    shutil.rmtree(target_dir)

                # Move extracted directory to final location
                shutil.move(str(source_dir), str(target_dir))

                return target_dir

        except (zipfile.BadZipFile, tarfile.TarError) as e:
            msg = f"Failed to extract {archive_path}: {e}"
            raise ExtractionError(msg) from e
        except OSError as e:
            msg = f"File system error during extraction: {e}"
            raise ExtractionError(msg) from e

    def _verify_checksum(self, file_path: Path, expected_sha256: str) -> None:
        """Verify file checksum.

        Parameters
        ----------
        file_path : Path
            Path to the file to verify.
        expected_sha256 : str
            Expected SHA-256 hash.

        Raises
        ------
        DownloadError
            If checksum doesn't match.
        """
        sha256_hash = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)

        actual_hash = sha256_hash.hexdigest()
        if actual_hash != expected_sha256:
            msg = (
                f"Checksum mismatch for {file_path}: expected {expected_sha256}, got {actual_hash}"
            )
            raise DownloadError(msg)


class VerbNetDownloader(BaseDownloader):
    """Downloads VerbNet from GitHub repository.

    Downloads the VerbNet dataset from the official GitHub repository
    using a specific commit hash for reproducibility.

    Attributes
    ----------
    dataset_name : str
        "verbnet"
    version : str
        "3.4"
    commit_hash : str
        "ae8e9cfdc2c0d3414b748763612f1a0a34194cc1"
    """

    @property
    def dataset_name(self) -> str:
        """Name of the dataset."""
        return "verbnet"

    @property
    def version(self) -> str:
        """Version of VerbNet."""
        return "3.4"

    @property
    def commit_hash(self) -> str:
        """GitHub repository commit hash."""
        return "ae8e9cfdc2c0d3414b748763612f1a0a34194cc1"

    def download(self, output_dir: Path) -> Path:
        """Download VerbNet dataset.

        Parameters
        ----------
        output_dir : Path
            Directory to download VerbNet to.

        Returns
        -------
        Path
            Path to the extracted VerbNet directory.

        Raises
        ------
        DownloadError
            If download fails.
        ExtractionError
            If extraction fails.
        """
        url = f"https://github.com/cu-clear/verbnet/archive/{self.commit_hash}.zip"
        archive_name = f"verbnet-{self.version}.zip"
        archive_path = output_dir / archive_name

        self._download_file(url, archive_path)

        try:
            extracted_dir = self._extract_archive(archive_path, output_dir)
        except ExtractionError:
            # Clean up failed download
            if archive_path.exists():
                archive_path.unlink()
            raise
        else:
            # Clean up archive file
            archive_path.unlink()
            return extracted_dir


class PropBankDownloader(BaseDownloader):
    """Downloads PropBank from GitHub repository.

    Downloads the PropBank frames from the official GitHub repository
    using a specific commit hash for reproducibility.

    Attributes
    ----------
    dataset_name : str
        "propbank"
    version : str
        "3.4.0"
    commit_hash : str
        "7280a04806b6ca3955ec82e28c4df96b6da76aef"
    """

    @property
    def dataset_name(self) -> str:
        """Name of the dataset."""
        return "propbank"

    @property
    def version(self) -> str:
        """Version of PropBank."""
        return "3.4.0"

    @property
    def commit_hash(self) -> str:
        """GitHub repository commit hash."""
        return "7280a04806b6ca3955ec82e28c4df96b6da76aef"

    def download(self, output_dir: Path) -> Path:
        """Download PropBank dataset.

        Parameters
        ----------
        output_dir : Path
            Directory to download PropBank to.

        Returns
        -------
        Path
            Path to the extracted PropBank directory.

        Raises
        ------
        DownloadError
            If download fails.
        ExtractionError
            If extraction fails.
        """
        url = f"https://github.com/propbank/propbank-frames/archive/{self.commit_hash}.zip"
        archive_name = f"propbank-{self.version}.zip"
        archive_path = output_dir / archive_name

        self._download_file(url, archive_path)

        try:
            extracted_dir = self._extract_archive(archive_path, output_dir)
        except ExtractionError:
            # Clean up failed download
            if archive_path.exists():
                archive_path.unlink()
            raise
        else:
            # Clean up archive file
            archive_path.unlink()
            return extracted_dir


class WordNetDownloader(BaseDownloader):
    """Downloads WordNet 3.1 from Princeton University.

    Downloads the WordNet 3.1 database from the official Princeton
    University distribution site.

    Attributes
    ----------
    dataset_name : str
        "wordnet"
    version : str
        "3.1"
    """

    @property
    def dataset_name(self) -> str:
        """Name of the dataset."""
        return "wordnet"

    @property
    def version(self) -> str:
        """Version of WordNet."""
        return "3.1"

    def download(self, output_dir: Path) -> Path:
        """Download WordNet dataset.

        Parameters
        ----------
        output_dir : Path
            Directory to download WordNet to.

        Returns
        -------
        Path
            Path to the extracted WordNet directory.

        Raises
        ------
        DownloadError
            If download fails.
        ExtractionError
            If extraction fails.
        """
        url = "https://wordnetcode.princeton.edu/wn3.1.dict.tar.gz"
        archive_name = "wordnet-3.1.tar.gz"
        archive_path = output_dir / archive_name

        self._download_file(url, archive_path)

        # WordNet tar.gz contains a 'dict' folder with all the data files
        try:
            # Extract to temp location first to see structure
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Extract archive
                with tarfile.open(str(archive_path), "r:gz") as tar_ref:
                    tar_ref.extractall(temp_path, filter="data")

                # The archive contains a 'dict' folder
                extracted_dict = temp_path / "dict"
                if not extracted_dict.exists():
                    raise ExtractionError("Expected 'dict' folder in WordNet archive")

                # Move to final location
                final_dict = output_dir / "wn31-dict"
                if final_dict.exists():
                    shutil.rmtree(final_dict)
                shutil.move(str(extracted_dict), str(final_dict))

                # Clean up archive file
                archive_path.unlink()
                return final_dict

        except (tarfile.TarError, OSError) as e:
            # Clean up failed download
            if archive_path.exists():
                archive_path.unlink()
            msg = f"Failed to extract WordNet archive: {e}"
            raise ExtractionError(msg) from e


class FrameNetDownloader(BaseDownloader):
    """Downloads FrameNet from NLTK data repository.

    Downloads FrameNet v1.7 from the NLTK data GitHub repository,
    which provides the dataset without license restrictions.

    Attributes
    ----------
    dataset_name : str
        "framenet"
    version : str
        "1.7"
    commit_hash : str
        "427fc05d3a8cc1ca99e7ff93bdea937507cc9e7a"
    """

    @property
    def dataset_name(self) -> str:
        """Name of the dataset."""
        return "framenet"

    @property
    def version(self) -> str:
        """Version of FrameNet."""
        return "1.7"

    @property
    def commit_hash(self) -> str:
        """NLTK data repository commit hash."""
        return "427fc05d3a8cc1ca99e7ff93bdea937507cc9e7a"

    def download(self, output_dir: Path) -> Path:
        """Download FrameNet from NLTK data repository.

        Parameters
        ----------
        output_dir : Path
            Directory to download FrameNet into.

        Returns
        -------
        Path
            Path to the extracted FrameNet directory.

        Raises
        ------
        DownloadError
            If download fails.
        ExtractionError
            If extraction fails.
        """
        url = f"https://raw.githubusercontent.com/nltk/nltk_data/{self.commit_hash}/packages/corpora/framenet_v17.zip"
        archive_path = output_dir / f"framenet-{self.version}.zip"

        try:
            print(f"Downloading {self.dataset_name} v{self.version}...")
            self._download_file(url, archive_path)

            print(f"Extracting {archive_path.name}...")
            extracted_path = self._extract_archive(archive_path, output_dir)

        except Exception as e:
            if isinstance(e, DownloadError | ExtractionError):
                raise
            msg = f"Failed to download {self.dataset_name}: {e}"
            raise DownloadError(msg) from e

        else:
            # Clean up archive on success
            if archive_path.exists():
                archive_path.unlink()
            return extracted_path

        finally:
            # Clean up archive on any exception
            if archive_path.exists():
                archive_path.unlink()


# Type alias for downloader classes
type DownloaderClass = (
    type[VerbNetDownloader]
    | type[PropBankDownloader]
    | type[WordNetDownloader]
    | type[FrameNetDownloader]
)

# Registry mapping dataset names to downloader classes
_DOWNLOADERS: dict[DatasetType, DownloaderClass] = {
    "verbnet": VerbNetDownloader,
    "propbank": PropBankDownloader,
    "wordnet": WordNetDownloader,
    "framenet": FrameNetDownloader,
}


def get_downloader(dataset: DatasetType | str) -> BaseDownloader:
    """Get downloader instance for a dataset.

    Parameters
    ----------
    dataset : DatasetType | str
        Name of the dataset to get downloader for (case-insensitive).

    Returns
    -------
    BaseDownloader
        Downloader instance for the specified dataset.

    Raises
    ------
    ValueError
        If dataset is not supported.

    Examples
    --------
    >>> downloader = get_downloader("verbnet")
    >>> print(downloader.version)
    3.4
    """
    # Normalize to lowercase for case-insensitive lookup. The cast narrows the
    # plain ``str`` to the key type for the lookup; ``.get`` still returns None
    # for any unsupported name, so the cast never asserts a false membership.
    dataset_lower = cast(DatasetType, dataset.lower())
    downloader_class = _DOWNLOADERS.get(dataset_lower)
    if downloader_class is None:
        supported = ", ".join(_DOWNLOADERS.keys())
        msg = f"Unsupported dataset: {dataset}. Supported: {supported}"
        raise ValueError(msg)
    return downloader_class()


def download_dataset(dataset: DatasetType | str, output_dir: Path) -> Path:
    """Download a specific dataset.

    Parameters
    ----------
    dataset : DatasetType | str
        Name of the dataset to download (case-insensitive).
    output_dir : Path
        Directory to download the dataset to.

    Returns
    -------
    Path
        Path to the downloaded dataset directory.

    Raises
    ------
    ValueError
        If dataset is not supported.
    DownloadError
        If download fails.
    ExtractionError
        If extraction fails.
    NotImplementedError
        If dataset requires manual download (FrameNet).

    Examples
    --------
    >>> from pathlib import Path
    >>> path = download_dataset("verbnet", Path("data/raw"))
    >>> print(f"Downloaded to: {path}")
    """
    downloader = get_downloader(dataset)
    return downloader.download(output_dir)


def download_all(
    output_dir: Path,
    datasets: list[DatasetType] | None = None,
) -> dict[DatasetType, Path | Exception]:
    """Download all available datasets.

    Parameters
    ----------
    output_dir : Path
        Directory to download datasets to.
    datasets : list[DatasetType] | None, default=None
        List of datasets to download. If None, downloads all supported datasets.

    Returns
    -------
    dict[DatasetType, Path | Exception]
        Mapping of dataset names to either the download path (success)
        or the exception that occurred (failure).

    Examples
    --------
    >>> from pathlib import Path
    >>> results = download_all(Path("data/raw"))
    >>> for dataset, result in results.items():
    ...     if isinstance(result, Path):
    ...         print(f"{dataset}: success -> {result}")
    ...     else:
    ...         print(f"{dataset}: failed -> {result}")
    """
    if datasets is None:
        datasets = list(_DOWNLOADERS.keys())

    results: dict[DatasetType, Path | Exception] = {}

    for dataset in datasets:
        try:
            path = download_dataset(dataset, output_dir)
            results[dataset] = path
            print(f"✓ {dataset}: {path}")
        except (DownloadError, ExtractionError, NotImplementedError) as e:
            results[dataset] = e
            print(f"✗ {dataset}: {e}")

    return results


def get_available_datasets() -> list[DatasetType]:
    """Get list of available datasets for download.

    Returns
    -------
    list[DatasetType]
        List of supported dataset names.

    Examples
    --------
    >>> datasets = get_available_datasets()
    >>> print(datasets)
    ['VerbNet', 'PropBank', 'WordNet', 'FrameNet']
    """
    return list(_DOWNLOADERS.keys())


def get_dataset_info(dataset: DatasetType | str) -> dict[str, str]:
    """Get information about a dataset.

    Parameters
    ----------
    dataset : DatasetType | str
        Name of the dataset (case-insensitive).

    Returns
    -------
    dict[str, str]
        Dictionary with dataset information including name and version.

    Raises
    ------
    ValueError
        If dataset is not supported.

    Examples
    --------
    >>> info = get_dataset_info("verbnet")
    >>> print(info["version"])
    3.4
    """
    downloader = get_downloader(dataset)
    return {
        "name": downloader.dataset_name,
        "version": downloader.version,
        "class": downloader.__class__.__name__,
    }
