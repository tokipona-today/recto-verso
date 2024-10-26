# utils.py
# à ajouter : gestion de la mémoire...

import json
import cv2
from pathlib import Path
import time
from functools import wraps
from datetime import datetime
import gc
import psutil
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple
from image_processing import ImageProcessor


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


def manage_memory(threshold_percent: float = 80.0,
                  force_gc: bool = True,
                  clear_cache: bool = True,
                  session_state: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Gère la mémoire de l'application en surveillant son utilisation et en libérant les ressources si nécessaire.

    Args:
        threshold_percent: Pourcentage de mémoire utilisée déclenchant le nettoyage
        force_gc: Si True, force l'exécution du garbage collector
        clear_cache: Si True, nettoie les caches d'images
        session_state: État de session Streamlit (optionnel)

    Returns:
        Dict contenant les statistiques de mémoire avant/après nettoyage
    """
    try:
        # Obtenir l'utilisation de la mémoire initiale
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # En MB
        system_memory_percent = psutil.virtual_memory().percent

        memory_stats = {
            'initial_memory_mb': initial_memory,
            'system_memory_percent': system_memory_percent,
            'cleaned_memory_mb': 0,
            'memory_saved_mb': 0
        }

        # Vérifier si le nettoyage est nécessaire
        if system_memory_percent > threshold_percent:
            logger.warning(f"Memory usage ({system_memory_percent}%) exceeds threshold ({threshold_percent}%)")

            # 1. Nettoyage du cache Streamlit si disponible
            if clear_cache and session_state is not None:
                if 'processor' in session_state and hasattr(session_state.processor, 'cache'):
                    session_state.processor.cache.clear()
                    logger.info("Cleared image processor cache")

                if 'current_results' in session_state:
                    # Garder uniquement le résultat final et la superposition
                    needed_results = ['final', 'overlay']
                    session_state.current_results = {
                        k: v for k, v in session_state.current_results.items()
                        if k in needed_results
                    }
                    logger.info("Cleared intermediate results")

            # 2. Forcer le garbage collector si demandé
            if force_gc:
                gc.collect()
                logger.info("Forced garbage collection")

            # 3. Nettoyer spécifiquement les objets numpy en mémoire
            for obj in gc.get_objects():
                if isinstance(obj, np.ndarray):
                    if not hasattr(obj, '_deprecated_shape'):  # Vérifier si l'array n'est pas déjà libéré
                        del obj

            # Mesurer la mémoire après nettoyage
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_saved = initial_memory - final_memory

            memory_stats.update({
                'cleaned_memory_mb': final_memory,
                'memory_saved_mb': max(0, memory_saved)  # Éviter les valeurs négatives
            })

            logger.info(f"Memory cleanup completed: {memory_saved:.2f}MB freed")

        return memory_stats

    except Exception as e:
        logger.error(f"Error during memory management: {str(e)}")
        return memory_stats


def get_memory_usage() -> Tuple[float, float]:
    """
    Retourne l'utilisation actuelle de la mémoire.

    Returns:
        Tuple contenant (utilisation en MB, pourcentage d'utilisation)
    """
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        memory_percent = psutil.virtual_memory().percent
        return memory_mb, memory_percent
    except Exception as e:
        logger.error(f"Error getting memory usage: {str(e)}")
        return 0.0, 0.0


def cleanup_numpy_cache():
    """
    Nettoie spécifiquement le cache numpy et libère la mémoire.
    """
    try:
        # Vider le cache numpy
        np.clear_numarray_cache()
        np.clear_typedict_cache()

        # Forcer la libération mémoire
        gc.collect()
        logger.info("Numpy cache cleaned")
    except Exception as e:
        logger.error(f"Error cleaning numpy cache: {str(e)}")


def optimize_image_loading(image_path: str, max_size: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Charge une image de manière optimisée avec gestion de la mémoire.

    Args:
        image_path: Chemin vers l'image
        max_size: Taille maximale optionnelle pour le redimensionnement

    Returns:
        Image chargée et optimisée ou None en cas d'erreur
    """
    try:
        # Lecture de l'image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image: {image_path}")
            return None

        # Conversion en RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Redimensionnement si nécessaire
        if max_size is not None:
            img = resize_image_if_needed(img, max_size)

        # Nettoyage préventif
        gc.collect()

        return img

    except Exception as e:
        logger.error(f"Error optimizing image: {str(e)}")
        return None


def clear_memory_intensive_operation(func):
    """
    Décorateur pour nettoyer la mémoire avant et après une opération intensive.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Nettoyage avant l'opération
            manage_memory(threshold_percent=70.0, force_gc=True)

            # Exécution de la fonction
            result = func(*args, **kwargs)

            # Nettoyage après l'opération
            manage_memory(threshold_percent=70.0, force_gc=True)

            return result

        except Exception as e:
            logger.error(f"Error in memory intensive operation: {str(e)}")
            raise

    return wrapper


def monitor_memory_usage():
    """
    Retourne une chaîne formatée avec l'utilisation actuelle de la mémoire.
    """
    try:
        memory_mb, memory_percent = get_memory_usage()
        return f"Memory Usage: {memory_mb:.1f}MB ({memory_percent:.1f}%)"
    except Exception as e:
        logger.error(f"Error monitoring memory: {str(e)}")
        return "Memory monitoring unavailable"


def cleanup_session_state(session_state: Dict[str, Any]) -> None:
    """
    Nettoie l'état de session Streamlit de manière sécurisée.

    Args:
        session_state: État de session Streamlit
    """
    try:
        keys_to_clear = []
        for key in session_state:
            if isinstance(session_state[key], (np.ndarray, ImageProcessor)):
                keys_to_clear.append(key)

        for key in keys_to_clear:
            del session_state[key]

        gc.collect()
        logger.info("Session state cleaned")

    except Exception as e:
        logger.error(f"Error cleaning session state: {str(e)}")


def get_memory_threshold_warning() -> Optional[str]:
    """
    Retourne un avertissement si l'utilisation de la mémoire est trop élevée.
    """
    try:
        _, memory_percent = get_memory_usage()
        if memory_percent > 90:
            return "⚠️ Utilisation mémoire critique. Envisagez de redémarrer l'application."
        elif memory_percent > 75:
            return "⚠️ Utilisation mémoire élevée"
        return None
    except Exception as e:
        logger.error(f"Error checking memory threshold: {str(e)}")
        return None


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
