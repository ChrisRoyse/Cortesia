# Task 01a: Install Neo4j Database

**Estimated Time**: 5 minutes  
**Dependencies**: None  
**Next Task**: 01b_verify_neo4j_connection.md  

## Objective
Install Neo4j database using Docker for development environment.

## Single Action
Install and start Neo4j database container.

## Command
```bash
# Pull and run Neo4j container
docker pull neo4j:5-community
docker run -d \
  --name llmkg-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/knowledge123 \
  -e NEO4J_dbms_memory_heap_max__size=2G \
  -e NEO4J_dbms_memory_pagecache_size=1G \
  neo4j:5-community
```

## Success Check
```bash
# Verify container is running
docker ps | grep llmkg-neo4j
# Should show: STATUS = Up X seconds
```

## Alternative (Non-Docker)
```bash
# For direct installation
# Download from: https://neo4j.com/download/
# Extract and run: bin/neo4j start
```

## Acceptance Criteria
- [ ] Neo4j container is running
- [ ] Ports 7474 and 7687 are accessible
- [ ] No error messages in docker logs

## Troubleshooting
If port conflicts occur:
```bash
docker run -d --name llmkg-neo4j -p 7475:7474 -p 7688:7687 ...
```

## Duration
2-5 minutes for download and startup.