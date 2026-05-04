# Pediatric Clinic AI Agent Workflow

Proof-of-concept Python workflow showing how multiple agents can triage, classify, route, draft responses, and log metrics for outpatient pediatric patient messages.

## Files

- `clinic_agents.py`: reusable agent classes and workflow coordinator
- `cli_demo.py`: command-line batch demo
- `streamlit_app.py`: lightweight dashboard
- `requirements.txt`: Streamlit dependency

## Run the CLI Demo

```bash
python3 cli_demo.py
```

## Run the Streamlit Dashboard

```bash
python3 -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## What It Demonstrates

The workflow processes each message through:

1. Message Intake Agent
2. Classification Agent
3. Risk Scoring Agent
4. Routing / Response Agent
5. Metrics & Logging Agent

It displays the cleaned message, classification, confidence score, matched terms, explainability statement, risk rationale, escalation triggers, assigned role, route, risk level, auto-send status, provider review flag, safety note, turnaround time, and final metrics summary.

## Robustness Improvements Added

- Added a separate Risk Scoring Agent
- Added symptom-combination escalation logic, such as fever + vomiting
- Added pediatric safety triggers, such as fever in a young infant
- Added explainability: "Why this classification?"
- Added escalation trigger display
- Added colored safety badges: auto-draft, staff review, provider review
- Added stronger business metrics: auto-draft percentage, review percentage, and time-savings assumptions
- Added more edge-case sample messages for demos

## Important Safety Note

This is a simulation only. It does not provide medical advice and should not be used for real patient care without clinical validation, EHR integration, security controls, HIPAA review, and human oversight.
