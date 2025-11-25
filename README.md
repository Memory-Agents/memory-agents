# Memory Augmented Agentic LLM: Initial Setup

This is the initial setup for the ADL project group 14: Memory Augmented Agentic LLM.

## Included Services

Currently the following services are included, we can modify and remove what we do not need as we go:

✅ Open WebUI - ChatGPT-like interface to privately interact with your local models and N8N agents

✅ Neo4j - Knowledge graph engine that powers tools like GraphRAG, LightRAG, and Graphiti

✅ Graphiti - A Graphiti submodule which provides the backend for

## Configuration

## Setup

**Requirements**
- Docker
- Python (for executing script)

**Copy environment vars**

WARNING: env.example SHOULD NEVER BE USED FOR PRODUCTION.

`cp .env.example .env`

**Start containers**

`` (cupython start_services.py --profile nonerrent setup)

**Open services**

- Openwebui (8080)
- ...