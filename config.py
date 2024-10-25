# config.py

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import os


@dataclass
class ImageProcessingParams:
    """Paramètres par défaut pour le traitement d'images"""
    # Prétraitement
    contrast: float = 2.4
    brightness: float = -38.0
    blur: int = 1
    denoise: int = 3

    # Analyse des gradients
    gradient_kernel: int = 3
    gradient_threshold: float = 1.5
    min_angle: float = 74.0
    max_angle: float = 90.0
    orientation_smoothing: int = 1
    magnitude_percentile: int = 70

    # Paramètres d'orientation
    orientation_morph_size: int = 3
    hysteresis_ratio: float = 0.5
    local_threshold_ratio: float = 0.95

    # Analyse des traits
    right_angle_threshold: float = 37.0
    left_angle_threshold: float = -30.0
    window_size: int = 21
    min_stroke_length: int = 20
    right_italic_weight: float = 0.15
    left_italic_weight: float = 0.3

    # Morphologie et pondération
    continuity_weight: float = 0.5
    intensity_weight: float = 0.3
    binary_threshold: float = 0.4
    remove_small_components: bool = True
    min_component_size: int = 70
    final_morphology: str = 'aucune'  # Ajout de la valeur par défaut
    final_kernel_size: int = 3
    final_iterations: int = 1
    smooth_edges: bool = True

    # Paramètres d'affichage
    opacity_preprocess: float = 0.3
    opacity_final: float = 0.5
    overlay_color: Tuple[int, int, int] = (255, 0, 0)  # Rouge par défaut


class Config:
    """Configuration générale de l'application"""

    # Chemins des fichiers
    DEFAULT_PARAMS_FILE = 'last_parameters.json'
    TEMP_DIR = 'temp'
    OUTPUT_DIR = 'output'

    # Configuration du traitement
    MAX_IMAGE_SIZE = 4000
    NUM_PROCESSES = os.cpu_count() or 4
    CHUNK_SIZE = 1000

    # Configuration de l'interface
    SIDEBAR_WIDTH = 350
    DEFAULT_IMAGE_WIDTH = 800

    # Extensions de fichiers supportées
    SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']

    @classmethod
    def ensure_directories(cls) -> None:
        """Crée les répertoires nécessaires s'ils n'existent pas"""
        for directory in [cls.TEMP_DIR, cls.OUTPUT_DIR]:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def is_supported_file(cls, filename: str) -> bool:
        """Vérifie si le fichier est dans un format supporté"""
        return os.path.splitext(filename)[1].lower() in cls.SUPPORTED_EXTENSIONS

    @classmethod
    def get_output_path(cls, input_path: str) -> str:
        """Génère le chemin de sortie pour un fichier traité"""
        basename = os.path.basename(input_path)
        name, ext = os.path.splitext(basename)
        return os.path.join(cls.OUTPUT_DIR, f"{name}_processed{ext}")