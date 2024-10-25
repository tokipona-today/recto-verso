# main.py

import streamlit as st
import numpy as np
import cv2
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from config import Config, ImageProcessingParams
from image_processing import ImageProcessor
from utils import (save_parameters, load_parameters, create_overlay_advanced,
                   resize_image_if_needed, format_timestamp)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_page_config():
    """Configure les paramètres de base de la page Streamlit"""
    st.set_page_config(
        page_title="Séparation des italiques",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def setup_page_style():
    """Configure le style CSS personnalisé de la page"""
    st.markdown(f"""
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {{
            width: {Config.SIDEBAR_WIDTH}px;
        }}
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {{
            width: {Config.SIDEBAR_WIDTH}px;
            margin-left: -{Config.SIDEBAR_WIDTH}px;
        }}
        .stButton>button {{
            width: 100%;
        }}
        .stProgress > div > div > div > div {{
            background-color: #1f77b4;
        }}
        .uploadedFile {{
            margin-bottom: 1rem;
        }}
        .stMarkdown {{
            font-size: 0.9rem;
        }}
        </style>
    """, unsafe_allow_html=True)


def setup_streamlit_page():
    """Configure la page Streamlit et son style"""
    setup_page_config()
    setup_page_style()


def init_session_state():
    """Initialise l'état de la session Streamlit"""
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'previous_params' not in st.session_state:
        st.session_state.previous_params = None
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'temp_image_path' not in st.session_state:
        st.session_state.temp_image_path = None


def create_sidebar_parameters() -> Dict[str, Any]:
    """
    Crée les widgets de la barre latérale pour les paramètres.

    Returns:
        Dict contenant les paramètres sélectionnés
    """
    st.sidebar.title("Paramètres")

    # Chargement des paramètres par défaut
    default_params = ImageProcessingParams()

    # Initialisation du dictionnaire de paramètres
    params = {}

    # Prétraitement
    st.sidebar.header("Prétraitement")
    params.update({
        'contrast': st.sidebar.slider(
            "Contraste",
            min_value=0.1,
            max_value=3.0,
            value=float(default_params.contrast),
            step=0.1
        ),
        'brightness': st.sidebar.slider(
            "Luminosité",
            min_value=-50,
            max_value=50,
            value=int(default_params.brightness),
            step=1
        ),
        'blur': st.sidebar.slider(
            "Flou",
            min_value=0,
            max_value=5,
            value=int(default_params.blur),
            step=1
        ),
        'denoise': st.sidebar.slider(
            "Débruitage",
            min_value=0,
            max_value=30,
            value=int(default_params.denoise),
            step=1
        )
    })

    # Analyse des gradients
    st.sidebar.header("Analyse des gradients")
    params.update({
        'gradient_kernel': st.sidebar.selectbox(
            "Taille noyau gradient",
            options=[3, 5, 7],
            index=0
        ),
        'gradient_threshold': st.sidebar.slider(
            "Seuil gradient",
            min_value=0.5,
            max_value=3.0,
            value=float(default_params.gradient_threshold),
            step=0.1
        ),
        'min_angle': st.sidebar.slider(
            "Angle minimum",
            min_value=-90.0,
            max_value=90.0,
            value=float(default_params.min_angle),
            step=0.1
        ),
        'max_angle': st.sidebar.slider(
            "Angle maximum",
            min_value=-90.0,
            max_value=90.0,
            value=float(default_params.max_angle),
            step=0.1
        ),
        'magnitude_percentile': st.sidebar.slider(
            "Percentile magnitude",
            min_value=50,
            max_value=90,
            value=int(default_params.magnitude_percentile),
            step=5
        ),
        'orientation_smoothing': st.sidebar.slider(
            "Lissage orientation",
            min_value=0,
            max_value=5,
            value=int(default_params.orientation_smoothing),
            step=1
        ),
        'orientation_morph_size': st.sidebar.slider(
            "Taille noyau morphologique",
            min_value=1,
            max_value=7,
            value=int(default_params.orientation_morph_size),
            step=2
        ),
        'hysteresis_ratio': st.sidebar.slider(
            "Ratio hystérésis",
            min_value=0.1,
            max_value=0.9,
            value=float(default_params.hysteresis_ratio),
            step=0.1
        ),
        'local_threshold_ratio': st.sidebar.slider(
            "Ratio seuil local",
            min_value=0.8,
            max_value=1.0,
            value=float(default_params.local_threshold_ratio),
            step=0.01
        )
    })

    # Analyse des traits
    st.sidebar.header("Analyse des traits")
    params.update({
        'right_angle_threshold': st.sidebar.slider(
            "Seuil angle droite",
            min_value=10,
            max_value=45,
            value=int(default_params.right_angle_threshold),
            step=1
        ),
        'left_angle_threshold': st.sidebar.slider(
            "Seuil angle gauche",
            min_value=-45,
            max_value=-10,
            value=int(default_params.left_angle_threshold),
            step=1
        ),
        'window_size': st.sidebar.slider(
            "Taille fenêtre d'analyse",
            min_value=11,
            max_value=31,
            value=int(default_params.window_size),
            step=2
        ),
        'min_stroke_length': st.sidebar.slider(
            "Longueur minimum traits",
            min_value=5,
            max_value=50,
            value=int(default_params.min_stroke_length),
            step=1
        )
    })

    # Pondération
    st.sidebar.header("Pondération")
    params.update({
        'right_italic_weight': st.sidebar.slider(
            "Poids italique droite",
            min_value=0.0,
            max_value=1.0,
            value=float(default_params.right_italic_weight),
            step=0.01
        ),
        'left_italic_weight': st.sidebar.slider(
            "Poids italique gauche",
            min_value=0.0,
            max_value=1.0,
            value=float(default_params.left_italic_weight),
            step=0.01
        ),
        'continuity_weight': st.sidebar.slider(
            "Poids continuité",
            min_value=0.0,
            max_value=1.0,
            value=float(default_params.continuity_weight),
            step=0.01
        ),
        'intensity_weight': st.sidebar.slider(
            "Poids intensité",
            min_value=0.0,
            max_value=1.0,
            value=float(default_params.intensity_weight),
            step=0.01
        ),
        'binary_threshold': st.sidebar.slider(
            "Seuil de binarisation",
            min_value=0.0,
            max_value=1.0,
            value=float(default_params.binary_threshold),
            step=0.01
        )
    })

    # Morphologie
    st.sidebar.header("Morphologie")
    params.update({
        'remove_small_components': st.sidebar.checkbox(
            "Supprimer petites composantes",
            value=default_params.remove_small_components
        ),
        'min_component_size': st.sidebar.slider(
            "Taille minimum composante",
            min_value=0,
            max_value=1000,
            value=int(default_params.min_component_size),
            step=10
        ),
        'final_morphology': st.sidebar.selectbox(
            "Opération morphologique finale",
            options=['aucune', 'ouverture', 'fermeture', 'dilatation', 'erosion'],
            index=0
        ),
        'final_kernel_size': st.sidebar.selectbox(
            "Taille noyau final",
            options=[3, 5, 7],
            index=0
        ),
        'final_iterations': st.sidebar.slider(
            "Itérations finales",
            min_value=0,
            max_value=5,
            value=int(default_params.final_iterations),
            step=1
        )
    })

    # Options finales
    st.sidebar.header("Finalisation")
    params.update({
        'smooth_edges': st.sidebar.checkbox(
            "Lisser les contours",
            value=default_params.smooth_edges
        )
    })

    # Paramètres de superposition
    st.sidebar.header("Superposition")
    params.update({
        'opacity_final': st.sidebar.slider(
            "Opacité résultat",
            min_value=0.0,
            max_value=1.0,
            value=float(default_params.opacity_final),
            step=0.1
        )
    })

    return params


def create_display_options() -> Dict[str, bool]:
    """
    Crée les options d'affichage dans la barre latérale droite.

    Returns:
        Dict contenant les options d'affichage
    """
    st.sidebar.header("Options d'affichage")

    display_options = {
        'original': st.sidebar.checkbox("Image originale", value=True),
        'preprocessed': st.sidebar.checkbox("Image prétraitée", value=True),
        # Ajout des masques manquants
        'continuity': st.sidebar.checkbox("Masque de continuité", value=True),
        'orientation': st.sidebar.checkbox("Masque d'orientation", value=True),
        'intensity': st.sidebar.checkbox("Masque d'intensité", value=True),
        'right_italic': st.sidebar.checkbox("Traits italiques droite", value=True),
        'left_italic': st.sidebar.checkbox("Traits italiques gauche", value=True),
        'weighted': st.sidebar.checkbox("Masque pondéré", value=True),
        'final': st.sidebar.checkbox("Résultat final", value=True),
        'overlay': st.sidebar.checkbox("Superposition finale", value=True)
    }

    return display_options


def handle_image_upload() -> Optional[str]:
    """
    Gère le téléchargement de l'image et sa sauvegarde temporaire.

    Returns:
        str: Chemin vers l'image temporaire ou None
    """
    uploaded_file = st.file_uploader(
        "Choisir une image",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        help="Formats supportés : PNG, JPEG, TIFF"
    )

    if uploaded_file is not None:
        if st.session_state.uploaded_image != uploaded_file:
            # Création du dossier temporaire si nécessaire
            temp_dir = Path(Config.TEMP_DIR)
            temp_dir.mkdir(exist_ok=True)

            # Sauvegarde du fichier temporaire
            file_extension = Path(uploaded_file.name).suffix
            temp_path = temp_dir / f"temp_image_{format_timestamp()}{file_extension}"

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Mise à jour de l'état de la session
            st.session_state.uploaded_image = uploaded_file

            # Suppression de l'ancien fichier temporaire s'il existe
            if st.session_state.temp_image_path:
                try:
                    Path(st.session_state.temp_image_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Erreur lors de la suppression du fichier temporaire: {e}")

            st.session_state.temp_image_path = str(temp_path)

            # Réinitialisation du processeur pour la nouvelle image
            st.session_state.processor = None
            st.session_state.current_results = None

            return str(temp_path)

        return st.session_state.temp_image_path

    return None


def process_image(file_path: str, params: Dict[str, Any]) -> None:
    """
    Traite l'image avec les paramètres spécifiés.

    Args:
        file_path: Chemin du fichier image
        params: Paramètres de traitement
    """
    try:
        # Création ou récupération du processeur d'image
        if st.session_state.processor is None:
            st.session_state.processor = ImageProcessor(file_path)

        # Traitement de l'image
        with st.spinner("Traitement de l'image en cours..."):
            results = st.session_state.processor.process_image(params)
            st.session_state.current_results = results

        # Sauvegarde des paramètres
        save_parameters(params)

    except Exception as e:
        logger.error(f"Erreur lors du traitement de l'image: {str(e)}")
        st.error(f"Une erreur est survenue lors du traitement: {str(e)}")


def display_results(results: Dict[str, np.ndarray],
                    display_options: Dict[str, bool],
                    params: Dict[str, Any]) -> None:
    """
    Affiche les résultats du traitement.

    Args:
        results: Résultats du traitement
        display_options: Options d'affichage
        params: Paramètres de traitement
    """
    if results is None:
        return

    # Configuration des colonnes
    col1, col2, col3 = st.columns([1, 10, 1])

    with col2:
        # Définition de l'ordre d'affichage des images
        image_order = [
            ('original', "Image originale"),
            ('preprocessed', "Image prétraitée"),
            ('continuity', "Masque de continuité"),
            ('orientation', "Masque d'orientation"),
            ('intensity', "Masque d'intensité"),
            ('right_italic', "Traits italiques droite"),
            ('left_italic', "Traits italiques gauche"),
            ('weighted', "Masque pondéré"),
            ('final', "Résultat final"),
            ('overlay', "Superposition finale")
        ]

        # Affichage des images selon les options
        current_row = []
        for i, (key, title) in enumerate(image_order):
            if display_options.get(key, False):
                if len(current_row) == 0:
                    cols = st.columns(2)  # Crée une nouvelle ligne avec 2 colonnes

                with cols[len(current_row)]:
                    st.subheader(title)
                    st.image(results[key], use_column_width=True)

                current_row.append(key)
                if len(current_row) == 2:  # Si la ligne est pleine
                    current_row = []  # Réinitialise pour la prochaine ligne
                    st.markdown("---")

        # Si il reste une image seule sur la dernière ligne
        if len(current_row) == 1:
            st.markdown("---")


def cleanup_temp_files():
    """Nettoie les fichiers temporaires anciens"""
    try:
        temp_dir = Path(Config.TEMP_DIR)
        current_time = time.time()
        for temp_file in temp_dir.glob("temp_image_*"):
            if current_time - temp_file.stat().st_mtime > 3600:  # 1 heure
                temp_file.unlink()
    except Exception as e:
        logger.warning(f"Erreur lors du nettoyage des fichiers temporaires: {e}")


def main():
    """Point d'entrée principal de l'application"""
    try:
        # Initialisation de la page
        setup_streamlit_page()
        init_session_state()

        # Création des dossiers nécessaires
        Config.ensure_directories()

        # Titre principal
        st.title("🔍 Séparation des italiques")

        # Zone de chargement de l'image
        col1, col2 = st.columns([2, 1])
        with col1:
            file_path = handle_image_upload()

        with col2:
            if file_path:
                st.success("Image chargée avec succès")
                if st.button("Réinitialiser", type="primary"):
                    st.session_state.uploaded_image = None
                    st.session_state.temp_image_path = None
                    st.session_state.processor = None
                    st.session_state.current_results = None
                    st.rerun()

        # Création des paramètres et options d'affichage
        params = create_sidebar_parameters()
        display_options = create_display_options()

        # Traitement de l'image
        if file_path and os.path.exists(file_path):
            # Affichage de l'image originale
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Vérification de la taille de l'image
            img = resize_image_if_needed(img, Config.MAX_IMAGE_SIZE)

            # Affichage des dimensions
            st.sidebar.markdown("---")
            st.sidebar.subheader("Informations image")
            st.sidebar.text(f"Dimensions: {img.shape[1]}x{img.shape[0]}")

            # Traitement si les paramètres ont changé
            if st.session_state.previous_params != params:
                process_image(file_path, params)
                st.session_state.previous_params = params.copy()

            # Affichage des résultats
            if st.session_state.current_results is not None:
                display_results(
                    st.session_state.current_results,
                    display_options,
                    params
                )

                # Bouton d'export
                st.sidebar.markdown("---")
                if st.sidebar.button("Exporter le résultat", type="primary"):
                    try:
                        output_path = Config.get_output_path(file_path)
                        if st.session_state.processor.save_result(
                                st.session_state.current_results['overlay'],
                                output_path
                        ):
                            st.sidebar.success(f"Résultat exporté: {output_path}")
                        else:
                            st.sidebar.error("Erreur lors de l'export")
                    except Exception as e:
                        st.sidebar.error(f"Erreur: {str(e)}")

        else:
            # Message d'instruction initial (suite)
            st.markdown("""
                            ### Instructions
                            1. Utilisez le bouton ci-dessus pour charger une image
                            2. Ajustez les paramètres dans la barre latérale
                            3. Visualisez les résultats en temps réel
                            4. Exportez l'image traitée quand vous êtes satisfait

                            ### Formats supportés
                            - PNG
                            - JPEG
                            - TIFF

                            ### Paramètres provisoirement recommandés (projet stradivarius)
                            - Contraste: 2.4
                            - Luminosité: -38
                            - Flou: 1
                            - Débruitage: 3
                            - Seuil gradient: 1.5
                            - Angles: 74° à 90°
                            - Seuil angle droite: 37°
                            - Taille fenêtre: 21
                            - Poids italique: 0.15 (droite), 0.3 (gauche)
                            - Taille minimum: 70 pixels
                        """)

    except Exception as e:
        logger.error(f"Erreur dans l'application principale: {str(e)}")
        st.error(f"Une erreur est survenue: {str(e)}")

    finally:
        # Nettoyage des fichiers temporaires
        cleanup_temp_files()

        # Message de débogage si nécessaire
        if os.environ.get('DEBUG'):
            st.sidebar.markdown("---")
            st.sidebar.subheader("État de la session")
            st.sidebar.json(st.session_state)

if __name__ == "__main__":
    main()
