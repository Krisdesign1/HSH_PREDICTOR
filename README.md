# HSH Predictor

Plateforme web de pronostics HSH avec backend automatisé.

Le produit fonctionne en deux couches :
- un service web FastAPI qui expose l'interface et les API de consultation ;
- un scheduler embarqué qui synchronise les matchs du jour, recalcule les pronostics et publie les résultats en base.

## Architecture

```text
FootyStats API
    ↓
collector.py        -> ingestion des ligues et matchs
    ↓
features.py         -> feature engineering
    ↓
model.py            -> clustering ligues + XGBoost calibré
    ↓
llm.py              -> ajustement contextuel optionnel
    ↓
value_bet.py        -> EV + Kelly
    ↓
automation.py       -> scheduler + publication automatique
    ↓
webapp.py           -> API / dashboard / admin
```

## Variables d'environnement

Copier `.env.example` vers `.env`, puis compléter :

```bash
FOOTYSTATS_API_KEY=
ANTHROPIC_API_KEY=
DATABASE_URL=postgresql://user:password@localhost:5432/hsh_predictor
```

Variables importantes :
- `FOOTYSTATS_API_KEY` : requise pour la synchronisation automatique.
- `ANTHROPIC_API_KEY` : optionnelle. Sans elle, le moteur LLM tombe en mode neutre.
- `DATABASE_URL` : requise.
- `AUTOMATION_ENABLED` : active le scheduler embarqué.
- `AUTOMATION_INTERVAL_SECONDS` : fréquence de recalcul.
- `TRACKED_LEAGUES` : filtre optionnel, liste séparée par virgules.

## Démarrage local

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

./.venv/bin/python main.py init
./.venv/bin/python main.py serve --host 127.0.0.1 --port 8000
```

Interface :
- `GET /` : frontend de consultation
- `GET /health` : healthcheck
- `GET /api/dashboard`
- `GET /api/predictions/today`
- `GET /api/predictions/history`
- `GET /api/admin/status`
- `POST /api/admin/run-now`

## Préparation des données

Le service web suppose qu'une base minimale et au moins un artefact modèle existent.

Bootstrap recommandé :

```bash
./.venv/bin/python main.py init
./.venv/bin/python main.py collect --seasons 3
./.venv/bin/python main.py cluster
./.venv/bin/python main.py train --group A
```

Notes :
- le service retombe sur `model_group_A.joblib` si le groupe demandé n'existe pas ;
- pour une qualité homogène, entraîner aussi les autres groupes disponibles.

## Déploiement Railway

Le repo contient déjà [railway.toml](./railway.toml). La stratégie actuelle est un service web unique avec scheduler in-process.
La version Python est épinglée via `.python-version` pour éviter le build par défaut en Python 3.13, incompatible avec les versions actuelles de `numpy` et `scikit-learn`.

### Étapes

1. Créer un projet Railway.
2. Ajouter un service PostgreSQL.
3. Déployer ce repo comme service web.
4. Définir les variables :

```bash
DATABASE_URL=${{Postgres.DATABASE_URL}}
FOOTYSTATS_API_KEY=...
ANTHROPIC_API_KEY=...
AUTOMATION_ENABLED=true
AUTOMATION_TRIGGER_ON_START=true
AUTOMATION_INTERVAL_SECONDS=1800
AUTOMATION_LOOKAHEAD_DAYS=0
AUTOMATION_MAX_LEAGUES=0
DEFAULT_BANKROLL=1000
```

Le démarrage utilisé par Railway est :

```bash
python main.py serve --host 0.0.0.0 --port $PORT
```

Healthcheck :

```text
/health
```

### Point d'attention

Les fichiers du dossier `models/` doivent être présents dans le déploiement. Si tu déploies depuis GitHub, il faut donc versionner les artefacts nécessaires ou prévoir une étape de génération séparée.

## Structure

```text
hsh_predictor/
├── automation.py
├── collector.py
├── config.py
├── database.py
├── features.py
├── llm.py
├── main.py
├── model.py
├── predictor.py
├── value_bet.py
├── webapp.py
├── static/
├── models/
└── railway.toml
```
