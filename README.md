# Memory Augmented Agentic LLM: Initial Setup

This is the initial setup for the ADL project group 14: Memory Augmented Agentic LLM.
This repo uses Docker and is built upon the Local AI Packaged repo, by coleam00: https://github.com/coleam00/local-ai-packaged.

## Included Services

Currently the following services are included, we can modify and remove what we do not need as we go:

✅ Self-hosted n8n - Low-code platform with over 400 integrations and advanced AI components

✅ Supabase - Open source database as a service - most widely used database for AI agents

✅ Ollama - Cross-platform LLM platform to install and run the latest local LLMs

✅ Open WebUI - ChatGPT-like interface to privately interact with your local models and N8N agents

✅ Neo4j - Knowledge graph engine that powers tools like GraphRAG, LightRAG, and Graphiti

✅ SearXNG - Open source, free internet metasearch engine which aggregates results from up to 229 search services. Users are neither tracked nor profiled, hence the fit with the local AI package.

✅ Langfuse - Open source LLM engineering platform for agent observability


## Configuration

The repo is currently configured to use a separate Ollama instance running on the host instead of in Docker directly:

```
x-n8n: &service-n8n
  image: n8nio/n8n:latest
  environment:
    - OLLAMA_HOST=host.docker.internal:11434
```

Can be adapted based on the hardware according to the docs: https://github.com/coleam00/local-ai-packaged.

## Setup

**Requirements**
- Docker
- Local Ollama installation (current setup)
- Python (for executing script)

**Copy environment vars**

WARNING: env.local SHOULD NEVER BE USED FOR PRODUCTION.

`cp .env.local .env`

**Start containers**

`python start_services.py --profile none` (current setup)

**Start Ollama**

`ollama serve`

**Open services**

- n8n (5678)
- Openwebui (8080)
- ...

## How to export n8n workflow files?

1. Click save for each workflow
2. Run `docker exec -it <container_id> n8n export:workflow --backup --output=.n8n/backup/workflows` in the root directory of this repository, substitute container id
  - You can find container id with `docker ps`
3. Commit to Git