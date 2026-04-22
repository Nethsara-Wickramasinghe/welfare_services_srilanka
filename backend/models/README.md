# Citizen Guidance Models

This directory stores the trained model artifacts used by the citizen-facing
welfare guidance backend.

Current files:

- `citizen_service_model.pkl`
- `citizen_priority_model.pkl`
- `citizen_model_metadata.json`

These artifacts are loaded by [backend/app.py](../app.py) to power
`POST /api/assess-citizen`.
