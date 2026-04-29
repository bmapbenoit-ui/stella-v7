# Handoff — Avis Google post-livraison via Gmail SMTP

## Contexte
Remplacement de FlowMail (limité par le plan gratuit) par un envoi via le SMTP Gmail (`info@planetebeauty.com`) déjà configuré sur Railway. Beaucoup de mails n'ont pas été envoyés depuis le 13/04/2026 — backfill prévu pour rattraper.

## Code livré (mergé sur `main`)
- **PR**: https://github.com/bmapbenoit-ui/stella-v7/pull/1 (mergée, squash)
- **Commit main**: `0e5bf76` — "feat: avis Google post-livraison via Gmail SMTP (#1)"
- **Branche source**: `claude/shopify-gmail-integration-VLcHp` (peut être supprimée)

### Endpoints ajoutés dans `stella-shopify-app/main.py`
1. `POST /api/webhook/google-review-request` — appelé par Shopify Flow
2. `POST /api/admin/google-review/backfill` — rattrapage manuel

### Auth
Header `X-Flow-Token` doit matcher l'env var `FLOW_WEBHOOK_TOKEN`.

### Table SQL créée à la volée
`google_review_emails(to_email, order_name, sent_at)` avec UNIQUE sur (to_email, order_name) — déduplication automatique.

## Variables Railway (déjà posées par l'utilisateur)
```
FLOW_WEBHOOK_TOKEN=3ebee7b98d1bfa5c454dcf58bd0e6cd06f3f89ad8d91d0100956ca9176d2d1ee
GOOGLE_REVIEW_URL=https://g.page/r/CTmV1GGiJ1rvEBM/review
```

## Shopify Flow (déjà reconfiguré par l'utilisateur)
Workflow `Demande d'avis post-livraison google` :
```
Fulfillment event created
       ↓
Condition: Fulfillment event status == DELIVERED → Vrai
       ↓
Wait 2 days
       ↓
Send HTTP request:
  POST https://stella-shopify-app-production.up.railway.app/api/webhook/google-review-request
  Headers: Content-Type: application/json
           X-Flow-Token: 3ebee7b98d1bfa5c454dcf58bd0e6cd06f3f89ad8d91d0100956ca9176d2d1ee
  Body: {"to": "{{ order.customer.email }}",
         "first_name": "{{ order.customer.firstName }}",
         "order_name": "{{ order.name }}"}
```

### Reste à faire côté Flow
- [ ] Passer `On client error (4XX)` de `Retry` à `Skip` (évite de spammer 401/400)
- [ ] **Activer le workflow** (badge "Arrêté" actuellement)

## ⚠️ BLOCKER actuel : Railway n'a pas redéployé

Symptôme : le backfill renvoie `{"detail":"Not Found"}`.

### Diagnostic à faire en priorité
```bash
# Test 1: l'app est-elle up ?
curl -i https://stella-shopify-app-production.up.railway.app/api/bis/smtp-status
# Attendu: 200 avec JSON. Si 404 → mauvaise URL ou app down.

# Test 2: le nouveau code est-il déployé ?
curl -i -X POST https://stella-shopify-app-production.up.railway.app/api/webhook/google-review-request \
  -H "Content-Type: application/json" -H "X-Flow-Token: wrong" -d '{}'
# Attendu si déployé: 401 Invalid token
# Si 404: redéploiement pas passé
```

### Actions selon le résultat
1. **Dashboard Railway → service `stella-shopify-app` → Deployments** : chercher le déploiement du commit `0e5bf76`
   - `Building/Deploying` → attendre
   - `Failed` → lire les logs, corriger
   - Aucun → vérifier `Settings → Source` (Railway suit-il bien `main` ?)
2. Si nécessaire, forcer un **Redeploy** depuis le dashboard

## Une fois Railway à jour : exécuter le backfill

### Dry-run (preview)
```bash
curl -X POST https://stella-shopify-app-production.up.railway.app/api/admin/google-review/backfill \
  -H "Content-Type: application/json" \
  -H "X-Flow-Token: 3ebee7b98d1bfa5c454dcf58bd0e6cd06f3f89ad8d91d0100956ca9176d2d1ee" \
  -d '{"since": "2026-04-13", "min_days_after_delivery": 2, "dry_run": true}'
```

Réponse attendue :
```json
{
  "ok": true, "dry_run": true,
  "fetched": N, "eligible": M,
  "skipped": {"no_email": ..., "no_delivery_date": ..., "too_recent": ...},
  "sample": [{"order_name": "#XXXX", "to": "...", "delivered_at": "..."}, ...]
}
```

### Envoi réel (retirer `dry_run`)
```bash
curl -X POST https://stella-shopify-app-production.up.railway.app/api/admin/google-review/backfill \
  -H "Content-Type: application/json" \
  -H "X-Flow-Token: 3ebee7b98d1bfa5c454dcf58bd0e6cd06f3f89ad8d91d0100956ca9176d2d1ee" \
  -d '{"since": "2026-04-13", "min_days_after_delivery": 2}'
```

Throttle 1 mail/sec côté serveur → si 200 mails, ça prend ~3-4 min.
Limite Gmail Workspace : 2000 mails/jour.

### Paramètres acceptés par le backfill
| Param | Type | Défaut | Effet |
|---|---|---|---|
| `since` | ISO date | `"2026-04-13"` | Borne inférieure `created_at` Shopify |
| `min_days_after_delivery` | int | `2` | Délai mini depuis `deliveredAt` |
| `dry_run` | bool | `false` | Liste sans envoyer |
| `limit` | int | `500` | Max commandes traitées par appel |

## Vérification post-envoi

```sql
-- Sur le Postgres Railway, table créée à la volée
SELECT COUNT(*), DATE(sent_at) FROM google_review_emails
GROUP BY DATE(sent_at) ORDER BY 2 DESC;
```

## Pour reprendre la main dans une session locale

L'agent peut continuer en lisant ce fichier. Lui dire :
> Lis HANDOFF_GOOGLE_REVIEW.md à la racine du repo et reprends le diagnostic
> Railway. Mon dernier test renvoyait `{"detail":"Not Found"}` sur le backfill.
