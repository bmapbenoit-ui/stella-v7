#!/bin/bash
# Script d'enrichissement BDK PARFUMS → Shopify
# Push les 29 metafields + SEO + description pour chaque produit
# Usage: bash push_shopify.sh

SHOP="planetemode.myshopify.com"
TOKEN="${SHOPIFY_ACCESS_TOKEN:?Définis SHOPIFY_ACCESS_TOKEN avant d'exécuter ce script}"
API_URL="https://${SHOP}/admin/api/2024-10/graphql.json"

push_product() {
  local JSON_FILE="$1"
  local PRODUCT_ID=$(jq -r '.shopify_id' "$JSON_FILE")
  local HANDLE=$(jq -r '.handle' "$JSON_FILE")
  local VENDOR=$(jq -r '.vendor' "$JSON_FILE")
  local PRODUCT_TYPE=$(jq -r '.productType' "$JSON_FILE")
  local SEO_TITLE=$(jq -r '.seo.title' "$JSON_FILE")
  local SEO_DESC=$(jq -r '.seo.description' "$JSON_FILE")
  local DESC_HTML=$(jq -r '.descriptionHtml' "$JSON_FILE")

  echo "================================================"
  echo "PUSH: $HANDLE"
  echo "ID:   $PRODUCT_ID"
  echo "================================================"

  # Build metafields array
  local METAFIELDS="["

  # Text single-line fields
  for field in concentration parfumeur genre accord_principal sillage tenacite longevite moment note_tete_principale note_coeur_principale note_fond_principale; do
    local val=$(jq -r ".metafields.${field}" "$JSON_FILE")
    METAFIELDS="${METAFIELDS}{\"namespace\":\"parfum\",\"key\":\"${field}\",\"value\":\"${val}\",\"type\":\"single_line_text_field\"},"
  done

  # Multi-line text fields
  for field in citation_parfumeur profil_porteur inci allergenes; do
    local val=$(jq -r ".metafields.${field}" "$JSON_FILE" | sed 's/"/\\"/g')
    METAFIELDS="${METAFIELDS}{\"namespace\":\"parfum\",\"key\":\"${field}\",\"value\":\"${val}\",\"type\":\"multi_line_text_field\"},"
  done

  # Integer fields
  for field in intensite sillage_level duree_tenue_heures contenance_ml annee_creation; do
    local val=$(jq -r ".metafields.${field}" "$JSON_FILE")
    if [ "$val" != "null" ] && [ -n "$val" ]; then
      METAFIELDS="${METAFIELDS}{\"namespace\":\"parfum\",\"key\":\"${field}\",\"value\":\"${val}\",\"type\":\"number_integer\"},"
    fi
  done

  # List fields (JSON arrays → list.single_line_text_field)
  for field in famille_olfactive notes_tete_secondaires notes_coeur_secondaires notes_fond_secondaires notes_cles accords_secondaires saison occasions attributs_speciaux; do
    local val=$(jq -c ".metafields.${field}" "$JSON_FILE")
    if [ "$val" != "null" ] && [ "$val" != "[]" ]; then
      METAFIELDS="${METAFIELDS}{\"namespace\":\"parfum\",\"key\":\"${field}\",\"value\":$(echo "$val" | jq -c '.' | jq -Rs .),\"type\":\"list.single_line_text_field\"},"
    fi
  done

  # Remove trailing comma and close array
  METAFIELDS="${METAFIELDS%,}]"

  # Escape descriptionHtml for JSON
  local DESC_ESCAPED=$(echo "$DESC_HTML" | jq -Rs '.')

  # Build the full GraphQL mutation
  local MUTATION=$(cat <<GRAPHQL
{
  "query": "mutation productUpdate(\$input: ProductInput!) { productUpdate(input: \$input) { product { id title vendor productType } userErrors { field message } } }",
  "variables": {
    "input": {
      "id": "${PRODUCT_ID}",
      "vendor": "${VENDOR}",
      "productType": "${PRODUCT_TYPE}",
      "descriptionHtml": ${DESC_ESCAPED},
      "seo": {
        "title": "${SEO_TITLE}",
        "description": "${SEO_DESC}"
      },
      "metafields": ${METAFIELDS}
    }
  }
}
GRAPHQL
)

  # Execute the mutation
  local RESPONSE=$(curl -s -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -H "X-Shopify-Access-Token: ${TOKEN}" \
    -d "$MUTATION")

  # Check for errors
  local USER_ERRORS=$(echo "$RESPONSE" | jq '.data.productUpdate.userErrors')
  local PRODUCT_RESULT=$(echo "$RESPONSE" | jq '.data.productUpdate.product')

  if [ "$USER_ERRORS" = "[]" ] && [ "$PRODUCT_RESULT" != "null" ]; then
    echo "✅ PUSH OK: $HANDLE"
    echo "   Product: $(echo "$PRODUCT_RESULT" | jq -r '.title')"
  else
    echo "❌ ERREUR PUSH: $HANDLE"
    echo "   UserErrors: $USER_ERRORS"
    echo "   Response: $RESPONSE"
    return 1
  fi

  echo ""
}

verify_product() {
  local PRODUCT_ID="$1"
  local HANDLE="$2"

  echo "🔍 Vérification post-push: $HANDLE"

  local QUERY=$(cat <<GRAPHQL
{
  "query": "query { product(id: \"${PRODUCT_ID}\") { id title vendor productType seo { title description } metafields(first: 50, namespace: \"parfum\") { edges { node { key value type } } } } }"
}
GRAPHQL
)

  local RESPONSE=$(curl -s -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -H "X-Shopify-Access-Token: ${TOKEN}" \
    -d "$QUERY")

  local METAFIELD_COUNT=$(echo "$RESPONSE" | jq '.data.product.metafields.edges | length')
  local VENDOR_CHECK=$(echo "$RESPONSE" | jq -r '.data.product.vendor')
  local SEO_TITLE_CHECK=$(echo "$RESPONSE" | jq -r '.data.product.seo.title')

  echo "   Metafields trouvés: ${METAFIELD_COUNT}/29"
  echo "   Vendor: ${VENDOR_CHECK}"
  echo "   SEO Title: ${SEO_TITLE_CHECK}"

  if [ "$METAFIELD_COUNT" -ge 25 ]; then
    echo "   ✅ Vérification OK"
  else
    echo "   ⚠️ Metafields incomplets (${METAFIELD_COUNT}/29)"
  fi
  echo ""
}

# ============================================
# MAIN: Push all 6 BDK products
# ============================================

echo "🚀 Début enrichissement BDK PARFUMS (6 produits)"
echo "Date: $(date)"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUCCESS=0
FAIL=0

for json_file in "$SCRIPT_DIR"/*.json; do
  if push_product "$json_file"; then
    ((SUCCESS++))
  else
    ((FAIL++))
  fi
done

echo "================================================"
echo "📊 RÉSULTAT PUSH: ${SUCCESS} OK / ${FAIL} ERREURS"
echo "================================================"
echo ""

# Verification pass
echo "🔍 Phase de vérification post-push..."
echo ""

for json_file in "$SCRIPT_DIR"/*.json; do
  PRODUCT_ID=$(jq -r '.shopify_id' "$json_file")
  HANDLE=$(jq -r '.handle' "$json_file")
  verify_product "$PRODUCT_ID" "$HANDLE"
done

echo "================================================"
echo "✅ Enrichissement BDK PARFUMS terminé"
echo "Date: $(date)"
echo "================================================"
