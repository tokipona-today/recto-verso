# image_processing.py

import numpy as np
import cv2
from scipy import ndimage
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.morphology import skeletonize
from typing import Dict, Any, Tuple, List
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Classe principale pour le traitement d'images"""

    def __init__(self, image_path: str):
        """
        Initialise le processeur d'images.

        Args:
            image_path: Chemin de l'image à traiter
        """
        self.image_path = image_path
        self.img_color = cv2.imread(image_path)
        if self.img_color is None:
            raise ValueError(f"Unable to read image: {image_path}")

        self.img_color = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2RGB)
        self.img_gray = cv2.cvtColor(self.img_color, cv2.COLOR_RGB2GRAY)
        self.original_shape = self.img_color.shape

        # Cache pour les résultats intermédiaires
        self.cache = {}
        self.param_versions = {
            'preprocess': 0,
            'gradients': 0,
            'stroke_detection': 0,
            'masks': 0,
            'morphology': 0,
            'final': 0
        }

        # Dépendances entre les paramètres et les étapes
        self.param_dependencies = {
            # Prétraitement
            'contrast': ['preprocess'],
            'brightness': ['preprocess'],
            'blur': ['preprocess'],
            'denoise': ['preprocess'],

            # Gradients
            'gradient_kernel': ['gradients'],
            'orientation_smoothing': ['gradients'],

            # Orientation et magnitude
            'gradient_threshold': ['masks'],
            'min_angle': ['masks'],
            'max_angle': ['masks'],
            'magnitude_percentile': ['masks', 'gradients'],
            'orientation_morph_size': ['masks'],
            'hysteresis_ratio': ['masks'],
            'local_threshold_ratio': ['masks'],

            # Détection des traits
            'right_angle_threshold': ['stroke_detection'],
            'left_angle_threshold': ['stroke_detection'],
            'window_size': ['stroke_detection'],
            'min_stroke_length': ['stroke_detection'],

            # Pondération
            'right_italic_weight': ['masks'],
            'left_italic_weight': ['masks'],
            'continuity_weight': ['masks'],
            'intensity_weight': ['masks'],
            'binary_threshold': ['masks'],

            # Morphologie
            'remove_small_components': ['morphology'],
            'min_component_size': ['morphology'],
            'final_morphology': ['morphology'],
            'final_kernel_size': ['morphology'],
            'final_iterations': ['morphology'],

            # Final
            'smooth_edges': ['final']
        }

        # Ordre des étapes de traitement
        self.processing_steps = [
            'preprocess',
            'gradients',
            'stroke_detection',
            'masks',
            'morphology',
            'final'
        ]

        self.last_result = None
        logger.info(f"Initialized ImageProcessor for {image_path}")

    def invalidate_cache(self, changed_params):
        """Invalide le cache pour les étapes affectées"""
        affected_steps = set()
        for param in changed_params:
            if param in self.param_dependencies:
                affected_steps.update(self.param_dependencies[param])

        # Invalide aussi toutes les étapes qui suivent
        all_affected = set()
        started_adding = False
        for step in self.processing_steps:
            if step in affected_steps:
                started_adding = True
            if started_adding:
                all_affected.add(step)
                if step in self.cache:
                    del self.cache[step]
                self.param_versions[step] += 1

    def preprocess_image(self, params):
        """Étape de prétraitement"""
        cache_key = ('preprocess', self.param_versions['preprocess'])
        if cache_key in self.cache:
            return self.cache[cache_key]

        img_preprocessed = self.img_gray.copy()
        img_preprocessed = cv2.convertScaleAbs(
            img_preprocessed,
            alpha=params['contrast'],
            beta=params['brightness']
        )

        if params['blur'] > 0:
            kernel_size = 2 * params['blur'] + 1
            img_preprocessed = cv2.GaussianBlur(
                img_preprocessed,
                (kernel_size, kernel_size),
                0
            )

        if params['denoise'] > 0:
            img_preprocessed = cv2.fastNlMeansDenoising(
                img_preprocessed.astype(np.uint8),
                None,
                params['denoise'],
                7,
                21
            )

        result = cv2.cvtColor(img_preprocessed, cv2.COLOR_GRAY2RGB)
        self.cache[cache_key] = result
        return result

    def compute_gradients(self, params, preprocessed):
        """Calcul des gradients"""
        cache_key = ('gradients', self.param_versions['gradients'])
        if cache_key in self.cache:
            return self.cache[cache_key]

        size = params['gradient_kernel']
        preprocessed_gray = cv2.GaussianBlur(
            cv2.cvtColor(preprocessed, cv2.COLOR_RGB2GRAY),
            (3, 3),
            0
        )

        # Utiliser Scharr pour une meilleure précision
        sobel_x = cv2.Scharr(preprocessed_gray, cv2.CV_64F, 1, 0)
        sobel_y = cv2.Scharr(preprocessed_gray, cv2.CV_64F, 0, 1)

        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        magnitude = cv2.GaussianBlur(magnitude, (5, 5), 0)

        orientation = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
        orientation = cv2.GaussianBlur(orientation, (3, 3), 0)

        result = {
            'sobel_x': sobel_x,
            'sobel_y': sobel_y,
            'magnitude': magnitude,
            'orientation': orientation
        }

        self.cache[cache_key] = result
        return result

    def detect_stroke_direction(self, preprocessed, params):
        """
        Détection améliorée des traits italiques avec meilleur filtrage
        """
        try:
            preprocessed_gray = cv2.cvtColor(preprocessed, cv2.COLOR_RGB2GRAY)
            window_size = params['window_size']
            angle_threshold_right = params['right_angle_threshold']
            angle_threshold_left = params['left_angle_threshold']

            right_italic_mask = np.zeros_like(preprocessed_gray, dtype=bool)
            left_italic_mask = np.zeros_like(preprocessed_gray, dtype=bool)

            # Amélioration du calcul des gradients avec Scharr
            grad_x = cv2.Scharr(preprocessed_gray, cv2.CV_64F, 1, 0)
            grad_y = cv2.Scharr(preprocessed_gray, cv2.CV_64F, 0, 1)

            gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            gradient_angle = np.arctan2(grad_y, grad_x) * 180 / np.pi

            # Seuil adaptatif plus strict
            grad_threshold = np.percentile(gradient_magnitude, 90)  # Augmentation du percentile
            strong_edges = gradient_magnitude > grad_threshold

            # Ajout d'un pré-filtrage pour éliminer le bruit
            strong_edges = cv2.morphologyEx(
                strong_edges.astype(np.uint8),
                cv2.MORPH_OPEN,
                np.ones((2, 2), np.uint8)
            ).astype(bool)

            step = window_size // 2  # Réduction du pas pour plus de précision

            for i in range(0, preprocessed_gray.shape[0] - window_size, step):
                for j in range(0, preprocessed_gray.shape[1] - window_size, step):
                    window_edges = strong_edges[i:i + window_size, j:j + window_size]
                    window_angles = gradient_angle[i:i + window_size, j:j + window_size]

                    # Augmentation du seuil de détection
                    if np.sum(window_edges) > window_size * 0.2:  # Plus strict
                        significant_angles = window_angles[window_edges]

                        # Histogramme plus fin
                        hist, bins = np.histogram(significant_angles, bins=90, range=(-180, 180))
                        smoothed_hist = ndimage.gaussian_filter1d(hist, sigma=1)

                        peak_threshold = np.max(smoothed_hist) * 0.6  # Plus strict
                        peaks = np.where(smoothed_hist > peak_threshold)[0]

                        for peak_idx in peaks:
                            angle = bins[peak_idx] + (bins[1] - bins[0]) / 2
                            if angle > 90:
                                angle -= 180
                            elif angle < -90:
                                angle += 180

                            strength = np.sum(window_edges) / window_edges.size
                            if strength > 0.25:  # Plus strict
                                if 15 < angle < angle_threshold_right:  # Plage d'angles plus stricte
                                    right_italic_mask[i:i + window_size, j:j + window_size] = True
                                elif angle_threshold_left < angle < -15:
                                    left_italic_mask[i:i + window_size, j:j + window_size] = True

            # Amélioration du post-traitement
            kernel = np.ones((2, 2), np.uint8)  # Kernel plus petit

            # Nettoyage initial
            right_italic_mask = cv2.morphologyEx(
                right_italic_mask.astype(np.uint8),
                cv2.MORPH_OPEN,
                kernel
            ).astype(bool)

            # Connexion des composantes proches
            right_italic_mask = cv2.morphologyEx(
                right_italic_mask.astype(np.uint8),
                cv2.MORPH_CLOSE,
                np.ones((3, 3), np.uint8)
            ).astype(bool)

            # Même traitement pour le masque gauche
            left_italic_mask = cv2.morphologyEx(
                left_italic_mask.astype(np.uint8),
                cv2.MORPH_OPEN,
                kernel
            ).astype(bool)
            left_italic_mask = cv2.morphologyEx(
                left_italic_mask.astype(np.uint8),
                cv2.MORPH_CLOSE,
                np.ones((3, 3), np.uint8)
            ).astype(bool)

            return right_italic_mask, left_italic_mask

        except Exception as e:
            logger.error(f"Error in detect_stroke_direction: {str(e)}")
            raise

    def detect_strokes(self, params, preprocessed):
        """
        Détection des traits
        """
        cache_key = ('stroke_detection', self.param_versions['stroke_detection'])
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Passage correct des paramètres à detect_stroke_direction
        right_italic, left_italic = self.detect_stroke_direction(
            preprocessed,
            params  # On passe tous les paramètres maintenant
        )

        enhanced_right = self.enhance_strokes(right_italic, params['min_stroke_length'])
        enhanced_left = self.enhance_strokes(left_italic, params['min_stroke_length'])

        result = {
            'right_italic': enhanced_right,
            'left_italic': enhanced_left
        }

        self.cache[cache_key] = result
        return result

    def enhance_strokes(self, mask, min_length=20):
        """Améliore la détection des traits"""
        skeleton = skeletonize(mask)
        nb_components, labels, stats, _ = cv2.connectedComponentsWithStats(
            skeleton.astype(np.uint8), connectivity=8)

        enhanced_mask = np.zeros_like(skeleton)
        for i in range(1, nb_components):
            if stats[i, cv2.CC_STAT_AREA] >= min_length:
                enhanced_mask[labels == i] = True

        return enhanced_mask

    def compute_masks(self, params, preprocessed, gradients, strokes):
        """
        Calcul amélioré des masques avec meilleure gestion de l'orientation
        """
        cache_key = ('masks', self.param_versions['masks'])
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 1. Préparation des données
        preprocessed_gray = cv2.cvtColor(preprocessed, cv2.COLOR_RGB2GRAY)

        # 2. Calcul du masque de continuité
        # Utilise la magnitude des gradients avec un seuil adaptatif
        magnitude_threshold = np.percentile(gradients['magnitude'], params['magnitude_percentile'])
        continuity_mask = (gradients['magnitude'] > magnitude_threshold * params['gradient_threshold']).astype(
            np.float32)

        # 3. Calcul du masque d'orientation amélioré
        # Normalisation des angles entre -90 et 90 degrés
        orientation = gradients['orientation'].copy()
        orientation = np.where(orientation > 90, orientation - 180, orientation)
        orientation = np.where(orientation < -90, orientation + 180, orientation)

        # Création du masque d'orientation avec plage d'angles
        orientation_mask = np.logical_and(
            orientation > params['min_angle'],
            orientation < params['max_angle']
        ).astype(np.float32)

        # Application d'un seuil sur la magnitude pour l'orientation
        orientation_mask = np.logical_and(
            orientation_mask,
            gradients['magnitude'] > magnitude_threshold
        ).astype(np.float32)

        # Amélioration de la continuité du masque d'orientation
        kernel_size = params['orientation_morph_size']
        orientation_mask = cv2.morphologyEx(
            orientation_mask,
            cv2.MORPH_CLOSE,
            np.ones((kernel_size, kernel_size), np.uint8)
        )

        # 4. Calcul du masque d'intensité
        # Utilisation d'un seuil adaptatif local
        local_mean = cv2.GaussianBlur(preprocessed_gray, (15, 15), 0)
        intensity_mask = (preprocessed_gray < (local_mean * params['local_threshold_ratio'])).astype(np.float32)

        # Amélioration du masque d'intensité
        intensity_mask = cv2.morphologyEx(
            intensity_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8)
        )

        # 5. Combinaison pondérée des masques
        # Normalisation des poids
        total_weight = (
                params['continuity_weight'] +
                params['right_italic_weight'] +
                params['left_italic_weight'] +
                params['intensity_weight']
        )

        # Calcul du masque pondéré
        weighted_mask = (
                                (params['continuity_weight'] * continuity_mask) +
                                (params['right_italic_weight'] * strokes['right_italic'].astype(np.float32)) +
                                (params['left_italic_weight'] * strokes['left_italic'].astype(np.float32)) +
                                (params['intensity_weight'] * intensity_mask)
                        ) / total_weight

        # 6. Binarisation avec hystérésis
        high_threshold = params['binary_threshold']
        low_threshold = high_threshold * params['hysteresis_ratio']

        # Création du masque initial avec double seuil
        strong_edges = weighted_mask > high_threshold
        weak_edges = np.logical_and(
            weighted_mask > low_threshold,
            weighted_mask <= high_threshold
        )

        # Connection des bords faibles aux bords forts
        labeled_strong, num_strong = ndimage.label(strong_edges)
        weak_labels = labeled_strong * weak_edges
        initial_mask = np.logical_or(strong_edges, weak_labels > 0)

        # 7. Post-traitement du masque final
        kernel = np.ones((3, 3), np.uint8)
        initial_mask = cv2.morphologyEx(
            initial_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel
        )

        # 8. Conversion finale et normalisation des masques
        result = {
            'continuity': (continuity_mask * 255).astype(np.uint8),
            'orientation': (orientation_mask * 255).astype(np.uint8),
            'intensity': (intensity_mask * 255).astype(np.uint8),
            'weighted': (weighted_mask * 255).astype(np.uint8),
            'initial': initial_mask.astype(np.uint8)
        }

        self.cache[cache_key] = result
        return result

    def apply_morphology(self, params, masks):
        """Application des opérations morphologiques"""
        cache_key = ('morphology', self.param_versions['morphology'])
        if cache_key in self.cache:
            return self.cache[cache_key]

        mask = masks['initial']

        if params['remove_small_components']:
            nb_components, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8)
            sizes = stats[1:, -1]
            min_size = params['min_component_size']

            clean_mask = np.zeros_like(mask)
            for i in range(1, nb_components):
                if sizes[i - 1] >= min_size:
                    clean_mask[labels == i] = True
        else:
            clean_mask = mask

        kernel = np.ones((params['final_kernel_size'], params['final_kernel_size']), np.uint8)

        if params['final_morphology'] == 'ouverture':
            result = cv2.morphologyEx(clean_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel,
                                      iterations=params['final_iterations'])
        elif params['final_morphology'] == 'fermeture':
            result = cv2.morphologyEx(clean_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel,
                                      iterations=params['final_iterations'])
        elif params['final_morphology'] == 'dilatation':
            result = cv2.dilate(clean_mask.astype(np.uint8), kernel,
                                iterations=params['final_iterations'])
        elif params['final_morphology'] == 'erosion':
            result = cv2.erode(clean_mask.astype(np.uint8), kernel,
                               iterations=params['final_iterations'])
        else:
            result = clean_mask.astype(np.uint8)

        self.cache[cache_key] = result
        return result

    def create_final_result(self, params, morphology_mask):
        """
        Création du résultat final

        Args:
            params: Paramètres de traitement
            morphology_mask: Masque après traitement morphologique

        Returns:
            Image finale traitée
        """
        cache_key = ('final', self.param_versions['final'])
        if cache_key in self.cache:
            return self.cache[cache_key]

        mask = morphology_mask.copy()
        if params['smooth_edges']:
            mask = cv2.GaussianBlur(mask.astype(float), (3, 3), 0)
            mask = mask > 0.5

        result = self.img_color.copy()
        result[~mask] = 255

        self.last_result = result
        self.cache[cache_key] = result
        return result

    def create_overlay_advanced(self,
                                original: np.ndarray,
                                preprocess: np.ndarray,
                                final: np.ndarray,
                                overlay_color: tuple,
                                opacity_preprocess: float,
                                opacity_final: float) -> np.ndarray:
        """
        Crée une superposition avancée avec contrôle d'opacité pour chaque couche

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

    def process_image(self, params: Dict[str, Any], previous_params: Dict[str, Any] = None) -> Dict[str, np.ndarray]:
        """
        Traitement principal avec gestion du cache

        Args:
            params: Paramètres actuels
            previous_params: Paramètres précédents pour la détection des changements

        Returns:
            Dictionnaire contenant les différents résultats du traitement
        """
        try:
            # Déterminer quels paramètres ont changé
            changed_params = set()
            if previous_params:
                changed_params = {k for k, v in params.items() if k in previous_params and v != previous_params[k]}
            else:
                changed_params = set(params.keys())

            # Invalider le cache pour les étapes affectées
            self.invalidate_cache(changed_params)

            with tqdm(total=6, desc="Processing image") as pbar:
                # Prétraitement
                preprocessed = self.preprocess_image(params)
                pbar.update(1)

                # Calcul des gradients
                gradients = self.compute_gradients(params, preprocessed)
                pbar.update(1)

                # Détection des traits
                strokes = self.detect_strokes(params, preprocessed)
                pbar.update(1)

                # Calcul des masques
                masks = self.compute_masks(params, preprocessed, gradients, strokes)
                pbar.update(1)

                # Application de la morphologie
                morphology = self.apply_morphology(params, masks)
                pbar.update(1)

                # Création du résultat final
                final = self.create_final_result(params, morphology)
                pbar.update(1)

                # Conversion des masques en images RGB pour l'affichage
                results = {
                    'original': self.img_color,
                    'preprocessed': preprocessed,
                    # Conversion correcte des masques en RGB
                    'continuity': cv2.cvtColor(masks['continuity'], cv2.COLOR_GRAY2RGB),
                    'orientation': cv2.cvtColor(masks['orientation'], cv2.COLOR_GRAY2RGB),
                    'intensity': cv2.cvtColor(masks['intensity'], cv2.COLOR_GRAY2RGB),
                    'right_italic': cv2.cvtColor((strokes['right_italic'] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB),
                    'left_italic': cv2.cvtColor((strokes['left_italic'] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB),
                    'weighted': cv2.cvtColor(masks['weighted'], cv2.COLOR_GRAY2RGB),
                    'final': final
                }

                # Ajout de la superposition
                overlay_color = (255, 0, 0)  # Rouge par défaut
                results['overlay'] = self.create_overlay_advanced(
                    results['original'],
                    results['preprocessed'],
                    results['final'],
                    overlay_color,
                    params.get('opacity_preprocess', 0.3),
                    params['opacity_final']
                )

                return results

        except Exception as e:
            logger.error(f"Error in process_image: {str(e)}")
            raise


    def save_result(self, result: np.ndarray, output_path: str) -> bool:
        """
        Sauvegarde le résultat du traitement

        Args:
            result: Image à sauvegarder
            output_path: Chemin de sauvegarde

        Returns:
            True si la sauvegarde a réussi, False sinon
        """
        try:
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr)
            logger.info(f"Result saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving result: {str(e)}")
            return False