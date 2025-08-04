# Task 01b: Verify Neo4j Connection

**Estimated Time**: 3 minutes  
**Dependencies**: 01a_install_neo4j.md  
**Next Task**: 01c_create_database_config.md  

## Objective
Test that Neo4j is accessible and responding to queries.

## Single Action
Execute a simple Cypher query to verify database connectivity.

## Test Commands
```bash
# Method 1: Using cypher-shell (if installed)
cypher-shell -a bolt://localhost:7687 -u neo4j -p knowledge123 "RETURN 'Hello Neo4j' as greeting"

# Method 2: Using curl to web interface
curl -H "Content-Type: application/json" \
     -X POST http://localhost:7474/db/data/cypher \
     -d '{"query":"RETURN 1 as test"}'
```

## Alternative Test (Docker exec)
```bash
# Execute cypher-shell inside container
docker exec llmkg-neo4j cypher-shell -u neo4j -p knowledge123 "RETURN 1"
```

## Expected Output
```
+---------------+
| greeting      |
+---------------+
| "Hello Neo4j" |
+---------------+
```

## Success Check
- [ ] Query executes without errors
- [ ] Returns expected result
- [ ] Connection established in <1 second

## Troubleshooting
```bash
# Check container logs
docker logs llmkg-neo4j

# Check if ports are listening
netstat -an | grep 7687
```

## Duration
1-3 minutes for connection verification.