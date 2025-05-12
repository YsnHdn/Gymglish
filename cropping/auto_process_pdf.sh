#!/bin/bash

# Script d'automatisation complète: magic-pdf + crop_from_middle.py avec support de plage de pages

# Définir la fonction d'aide
show_help() {
    echo "Usage: $0 <chemin_fichier_pdf> <dossier_output> [options]"
    echo ""
    echo "Options obligatoires:"
    echo "  <chemin_fichier_pdf>   Chemin vers le fichier PDF à traiter"
    echo "  <dossier_output>       Dossier où enregistrer les résultats"
    echo ""
    echo "Options facultatives:"
    echo "  -s, --start PAGE       Numéro de la première page à traiter (par défaut: 1)"
    echo "  -e, --end PAGE         Numéro de la dernière page à traiter (par défaut: dernière page)"
    echo "  -d, --dpi DPI          Résolution DPI pour le découpage (par défaut: 300)"
    echo "  -h, --help             Afficher cette aide"
    echo ""
    echo "Exemple:"
    echo "  $0 document.pdf ./sortie -s 5 -e 10 -d 400"
    echo "  Traite les pages 5 à 10 du document.pdf avec une résolution de 400 DPI"
}

# Vérifier le nombre minimal d'arguments
if [ "$#" -lt 2 ]; then
    show_help
    exit 1
fi

# Récupérer les deux premiers arguments obligatoires
PDF_INPUT="$1"
OUTPUT_DIR="$2"
shift 2  # Décaler les arguments pour traiter les options

# Valeurs par défaut
START_PAGE=""
END_PAGE=""
DPI=300

# Traiter les options
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--start)
            START_PAGE="$2"
            shift 2
            ;;
        -e|--end)
            END_PAGE="$2"
            shift 2
            ;;
        -d|--dpi)
            DPI="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Option non reconnue: $1"
            show_help
            exit 1
            ;;
    esac
done

# Vérifier si le fichier PDF existe
if [ ! -f "$PDF_INPUT" ]; then
    echo "❌ Fichier PDF non trouvé: $PDF_INPUT"
    exit 1
fi

# Extraire le nom du fichier sans extension
FILENAME=$(basename "$PDF_INPUT" .pdf)

# Créer les dossiers s'ils n'existent pas
mkdir -p "$OUTPUT_DIR"

# Construire la commande magic-pdf avec les options de plage de pages
MAGIC_PDF_CMD="magic-pdf -p \"$PDF_INPUT\" -o \"$OUTPUT_DIR\""

if [ -n "$START_PAGE" ]; then
    MAGIC_PDF_CMD="$MAGIC_PDF_CMD -s $START_PAGE"
fi

if [ -n "$END_PAGE" ]; then
    MAGIC_PDF_CMD="$MAGIC_PDF_CMD -e $END_PAGE"
fi

echo "⏳ Étape 1/2: Exécution de magic-pdf sur $PDF_INPUT..."
echo "   Commande: $MAGIC_PDF_CMD"

# Exécuter la commande magic-pdf
eval $MAGIC_PDF_CMD
MAGIC_PDF_EXIT_CODE=$?

# Vérifier si l'exécution a réussi
if [ $MAGIC_PDF_EXIT_CODE -ne 0 ]; then
    echo "❌ Erreur lors de l'exécution de magic-pdf (code de sortie: $MAGIC_PDF_EXIT_CODE)"
    exit 1
fi

echo "✅ Traitement magic-pdf terminé"

# Chemin des fichiers générés par magic-pdf
# Notez que magic-pdf crée un sous-dossier avec le nom du fichier PDF et la méthode d'analyse
MAGIC_PDF_OUTPUT_DIR="${OUTPUT_DIR}/${FILENAME}/auto"
LAYOUT_PDF="${MAGIC_PDF_OUTPUT_DIR}/${FILENAME}_layout.pdf"
MIDDLE_JSON="${MAGIC_PDF_OUTPUT_DIR}/${FILENAME}_middle.json"

# Vérifier si le dossier de sortie existe
if [ ! -d "$MAGIC_PDF_OUTPUT_DIR" ]; then
    echo "❌ Dossier de sortie de magic-pdf non trouvé: $MAGIC_PDF_OUTPUT_DIR"
    echo "   Vérifiez la structure des dossiers générés par magic-pdf."
    exit 1
fi

# Vérifier si les fichiers nécessaires existent
if [ ! -f "$LAYOUT_PDF" ]; then
    echo "❌ Fichier de mise en page non trouvé: $LAYOUT_PDF"
    # Rechercher où se trouve réellement le fichier layout.pdf
    echo "   Recherche du fichier layout.pdf dans le dossier de sortie..."
    LAYOUT_PDF_FOUND=$(find "$OUTPUT_DIR" -name "${FILENAME}_layout.pdf" -type f -print | head -n 1)
    
    if [ -n "$LAYOUT_PDF_FOUND" ]; then
        echo "   Fichier trouvé à: $LAYOUT_PDF_FOUND"
        LAYOUT_PDF="$LAYOUT_PDF_FOUND"
        # Mise à jour du chemin du JSON également
        MIDDLE_JSON_DIR=$(dirname "$LAYOUT_PDF_FOUND")
        MIDDLE_JSON="${MIDDLE_JSON_DIR}/${FILENAME}_middle.json"
    else
        echo "   Aucun fichier layout.pdf trouvé."
        exit 1
    fi
fi

if [ ! -f "$MIDDLE_JSON" ]; then
    echo "❌ Fichier JSON intermédiaire non trouvé: $MIDDLE_JSON"
    exit 1
fi

# Créer le dossier de sortie pour les crops
CROP_OUTPUT_DIR="${MAGIC_PDF_OUTPUT_DIR}/crops"
mkdir -p "$CROP_OUTPUT_DIR"

echo "⏳ Étape 2/2: Exécution de crop_from_middle.py..."

# Exécuter le script crop_from_middle.py
python crop_from_middle.py --pdf "$LAYOUT_PDF" --middle "$MIDDLE_JSON" --output "$CROP_OUTPUT_DIR" --dpi "$DPI"
CROP_EXIT_CODE=$?

# Vérifier si l'exécution a réussi
if [ $CROP_EXIT_CODE -ne 0 ]; then
    echo "❌ Erreur lors de l'exécution de crop_from_middle.py (code de sortie: $CROP_EXIT_CODE)"
    exit 1
fi

echo "✅ Traitement complet terminé avec succès!"
echo "   - Résultats de magic-pdf: $MAGIC_PDF_OUTPUT_DIR"
echo "   - Éléments découpés: $CROP_OUTPUT_DIR"
echo "   - Texte et équations: $CROP_OUTPUT_DIR/text_blocks"
echo "   - Équations en ligne: $CROP_OUTPUT_DIR/equations"
echo "   - Métadonnées complètes: $CROP_OUTPUT_DIR/all_content.json"

exit 0