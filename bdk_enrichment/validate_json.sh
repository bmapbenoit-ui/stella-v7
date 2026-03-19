#!/bin/bash
# Validation des 29 champs metafields avant push Shopify
# Usage: bash validate_json.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TEXT_FIELDS="concentration parfumeur genre accord_principal sillage tenacite longevite moment note_tete_principale note_coeur_principale note_fond_principale"
MULTILINE_FIELDS="citation_parfumeur profil_porteur inci allergenes"
INT_FIELDS="intensite sillage_level duree_tenue_heures contenance_ml annee_creation"
LIST_FIELDS="famille_olfactive notes_tete_secondaires notes_coeur_secondaires notes_fond_secondaires notes_cles accords_secondaires saison occasions attributs_speciaux"

TOTAL_OK=0
TOTAL_FAIL=0

for json_file in "$SCRIPT_DIR"/*.json; do
  HANDLE=$(jq -r '.handle' "$json_file")
  echo "================================================"
  echo "Validation: $HANDLE"
  echo "================================================"

  MISSING=0
  PRESENT=0

  # Check text fields
  for field in $TEXT_FIELDS; do
    val=$(jq -r ".metafields.${field} // empty" "$json_file")
    if [ -z "$val" ]; then
      echo "  ❌ MANQUANT: $field"
      ((MISSING++))
    else
      ((PRESENT++))
    fi
  done

  # Check multiline fields
  for field in $MULTILINE_FIELDS; do
    val=$(jq -r ".metafields.${field} // empty" "$json_file")
    if [ -z "$val" ]; then
      echo "  ❌ MANQUANT: $field"
      ((MISSING++))
    else
      ((PRESENT++))
    fi
  done

  # Check integer fields
  for field in $INT_FIELDS; do
    val=$(jq ".metafields.${field} // empty" "$json_file")
    if [ -z "$val" ] || [ "$val" = "null" ]; then
      echo "  ❌ MANQUANT: $field"
      ((MISSING++))
    else
      ((PRESENT++))
    fi
  done

  # Check list fields
  for field in $LIST_FIELDS; do
    val=$(jq ".metafields.${field} // empty" "$json_file")
    if [ -z "$val" ] || [ "$val" = "null" ] || [ "$val" = "[]" ]; then
      echo "  ❌ MANQUANT: $field"
      ((MISSING++))
    else
      ((PRESENT++))
    fi
  done

  # Check SEO
  seo_title=$(jq -r '.seo.title // empty' "$json_file")
  seo_desc=$(jq -r '.seo.description // empty' "$json_file")
  seo_title_len=${#seo_title}
  seo_desc_len=${#seo_desc}

  echo ""
  echo "  SEO Title (${seo_title_len} chars, max 70): $seo_title"
  if [ $seo_title_len -gt 70 ]; then
    echo "  ⚠️ SEO Title trop long!"
  fi
  echo "  SEO Desc (${seo_desc_len} chars, max 155): $seo_desc"
  if [ $seo_desc_len -gt 155 ]; then
    echo "  ⚠️ SEO Desc trop longue!"
  fi

  # Check descriptionHtml
  desc=$(jq -r '.descriptionHtml // empty' "$json_file")
  if [ -z "$desc" ]; then
    echo "  ❌ MANQUANT: descriptionHtml"
  fi

  # Check standardized values
  sillage_val=$(jq -r '.metafields.sillage' "$json_file")
  tenacite_val=$(jq -r '.metafields.tenacite' "$json_file")
  longevite_val=$(jq -r '.metafields.longevite' "$json_file")

  echo ""
  echo "  Sillage: $sillage_val"
  echo "  Ténacité: $tenacite_val"
  echo "  Longévité: $longevite_val"

  case "$sillage_val" in
    "Intime"|"Discret"|"Modéré"|"Modéré à fort"|"Fort"|"Très fort") ;;
    *) echo "  ⚠️ Sillage non standardisé!" ;;
  esac

  case "$tenacite_val" in
    "Courte tenue"|"Tenue modérée"|"Longue tenue"|"Très longue tenue") ;;
    *) echo "  ⚠️ Ténacité non standardisée!" ;;
  esac

  case "$longevite_val" in
    "Courte"|"Modérée"|"Longue"|"Très longue") ;;
    *) echo "  ⚠️ Longévité non standardisée!" ;;
  esac

  echo ""
  echo "  📊 Résultat: ${PRESENT}/29 champs présents, ${MISSING} manquants"

  if [ $MISSING -eq 0 ]; then
    echo "  ✅ VALIDE — prêt pour push"
    ((TOTAL_OK++))
  else
    echo "  ❌ NON VALIDE — ${MISSING} champs manquants"
    ((TOTAL_FAIL++))
  fi
  echo ""
done

echo "================================================"
echo "📊 TOTAL: ${TOTAL_OK} valides / ${TOTAL_FAIL} invalides sur $((TOTAL_OK + TOTAL_FAIL)) produits"
echo "================================================"

if [ $TOTAL_FAIL -eq 0 ]; then
  echo "✅ Tous les produits sont prêts pour le push Shopify!"
  echo "   Exécute: bash push_shopify.sh"
else
  echo "❌ Corrige les erreurs avant de push!"
fi
