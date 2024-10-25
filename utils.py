# utils.py

import json
import logging
from typing import Dict, Any, Optional
import cv2
import numpy as np
from pathlib import Path
import time
from functools import wraps
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def timing_decorator(func):
    """Décorateur pour mesurer le temps d'exécution des fonctions"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result

    return wrapper


def save_parameters(params: Dict[str, Any], filepath: str = 'last_parameters.json') -> None:
    """
    Sauvegarde les paramètres dans un fichier JSON.

    Args:
        params: Dictionnaire des paramètres à sauvegarder
        filepath: Chemin du fichier de sauvegarde
    """
    try:
        # Conversion des types non sérialisables
        clean_params = {}
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                clean_params[key] = value.tolist()
            elif isinstance(value, (np.int64, np.float64)):
                clean_params[key] = float(value)
            else:
                clean_params[key] = value

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(clean_params, f, indent=4)
        logger.info(f"Parameters successfully saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving parameters: {str(e)}")
        raise


def load_parameters(filepath: str = 'last_parameters.json') -> Optional[Dict[str, Any]]:
    """
    Charge les derniers paramètres utilisés.

    Args:
        filepath: Chemin du fichier de paramètres

    Returns:
        Dict contenant les paramètres ou None si erreur
    """
    try:
        if not Path(filepath).exists():
            logger.warning(f"Parameters file {filepath} not found")
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            params = json.load(f)
        logger.info(f"Parameters successfully loaded from {filepath}")
        return params
    except Exception as e:
        logger.error(f"Error loading parameters: {str(e)}")
        return None


def create_overlay_advanced(original: np.ndarray,
                            preprocess: np.ndarray,
                            final: np.ndarray,
                            overlay_color: tuple,
                            opacity_preprocess: float,
                            opacity_final: float) -> np.ndarray:
    """
    Crée une superposition avancée avec contrôle d'opacité pour chaque couche.

    Args:
        original: Image originale
        preprocess: Image prétraitée
        final: Masque final
        overlay_color: Couleur de superposition (R,G,B)
        opacity_preprocess: Opacité de la couche prétraitée
        opacity_final: Opacité de la couche finale

    Returns:
        Image avec superposition
    """
    try:
        if len(final.shape) == 3:
            mask = cv2.cvtColor(final, cv2.COLOR_RGB2GRAY)
        else:
            mask = final.copy()

        binary_mask = (mask < 128)
        result = original.copy()

        if opacity_preprocess > 0:
            result = cv2.addWeighted(
                result,
                1 - opacity_preprocess,
                preprocess,
                opacity_preprocess,
                0
            )

        color_layer = np.zeros_like(original)
        color_layer[binary_mask] = overlay_color

        if opacity_final > 0:
            overlay_mask = binary_mask.astype(np.float32) * opacity_final
            for i in range(3):
                result[..., i] = result[..., i] * (1 - overlay_mask) + \
                                 color_layer[..., i] * overlay_mask

        return result.astype(np.uint8)
    except Exception as e:
        logger.error(f"Error creating overlay: {str(e)}")
        raise


def format_timestamp() -> str:
    """Retourne un horodatage formaté pour les noms de fichiers"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resize_image_if_needed(image: np.ndarray, max_size: int = 4000) -> np.ndarray:
    """
    Redimensionne l'image si elle dépasse la taille maximale.

    Args:
        image: Image à redimensionner
        max_size: Taille maximale en pixels

    Returns:
        Image redimensionnée si nécessaire
    """
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image