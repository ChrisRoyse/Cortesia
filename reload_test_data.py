#!/usr/bin/env python3
"""
Reload test data to LLMKG database
"""

import requests
import json
import time

# Backend configuration
API_BASE_URL = "http://localhost:3001"

def store_triple(subject, predicate, obj, confidence=1.0):
    """Store a triple using the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/triple",
            headers={"Content-Type": "application/json"},
            json={
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "confidence": confidence
            }
        )
        
        if response.status_code == 200:
            print(f"SUCCESS: {subject} {predicate} {obj}")
            return response.json()
        else:
            print(f"ERROR: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def store_entity(name, entity_type, description="", properties=None):
    """Store an entity using the API"""
    if properties is None:
        properties = {}
        
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/entity",
            headers={"Content-Type": "application/json"},
            json={
                "name": name,
                "entity_type": entity_type,
                "description": description,
                "properties": properties
            }
        )
        
        if response.status_code == 200:
            print(f"SUCCESS: Entity {name} ({entity_type})")
            return response.json()
        else:
            print(f"ERROR: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def store_chunk(text):
    """Store a text chunk using the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/chunk",
            headers={"Content-Type": "application/json"},
            json={"text": text}
        )
        
        if response.status_code == 200:
            print(f"SUCCESS: Chunk stored")
            return response.json()
        else:
            print(f"ERROR: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def main():
    print("Reloading LLMKG Test Data")
    print("=" * 30)
    
    # Check if server is running
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/discovery", timeout=5)
        if response.status_code != 200:
            print("ERROR: LLMKG API server not responding")
            return
    except:
        print("ERROR: Cannot connect to LLMKG API server")
        return
    
    print("SUCCESS: Server is responding")
    
    # Add entities first
    print("\n1. Adding entities...")
    entities = [
        {"name": "Claude", "entity_type": "AI Assistant", "description": "AI assistant created by Anthropic"},
        {"name": "Anthropic", "entity_type": "Company", "description": "AI safety company"},
        {"name": "Transformer", "entity_type": "Architecture", "description": "Neural network architecture"},
        {"name": "Google", "entity_type": "Company", "description": "Technology company"}, 
        {"name": "San Francisco", "entity_type": "Location", "description": "City in California"},
        {"name": "California", "entity_type": "Location", "description": "State in United States"},
        {"name": "LLMKG", "entity_type": "System", "description": "Knowledge graph system"},
        {"name": "MCP", "entity_type": "Protocol", "description": "Model Context Protocol"},
        {"name": "Neural Network", "entity_type": "Technology", "description": "Machine learning model"},
        {"name": "Machine Learning", "entity_type": "Technology", "description": "AI technique"},
        {"name": "Artificial Intelligence", "entity_type": "Technology", "description": "Computer intelligence"},
    ]
    
    for entity in entities:
        store_entity(entity["name"], entity["entity_type"], entity["description"])
        time.sleep(0.1)
    
    print("\n2. Adding relationships (triples)...")
    facts = [
        {"subject": "Claude", "predicate": "is", "object": "AI assistant"},
        {"subject": "Claude", "predicate": "created_by", "object": "Anthropic"},
        {"subject": "Claude", "predicate": "uses", "object": "Transformer"},
        {"subject": "Anthropic", "predicate": "is", "object": "Company"},
        {"subject": "Anthropic", "predicate": "located_in", "object": "San Francisco"},
        {"subject": "Transformer", "predicate": "is", "object": "Neural Network"},
        {"subject": "Transformer", "predicate": "invented_by", "object": "Google"},
        {"subject": "Neural Network", "predicate": "is", "object": "Machine Learning"},
        {"subject": "Machine Learning", "predicate": "is_part_of", "object": "Artificial Intelligence"},
        {"subject": "San Francisco", "predicate": "is_in", "object": "California"},
        {"subject": "California", "predicate": "is_in", "object": "United States"},
        {"subject": "LLMKG", "predicate": "is", "object": "System"},
        {"subject": "LLMKG", "predicate": "uses", "object": "MCP"},
        {"subject": "MCP", "predicate": "enables", "object": "tool communication"},
        {"subject": "Google", "predicate": "developed", "object": "Transformer"},
        {"subject": "AI assistant", "predicate": "helps", "object": "users"},
        {"subject": "System", "predicate": "stores", "object": "knowledge"},
        {"subject": "knowledge", "predicate": "contains", "object": "facts"},
    ]
    
    for fact in facts:
        store_triple(fact["subject"], fact["predicate"], fact["object"])
        time.sleep(0.1)
    
    print("\n3. Adding knowledge chunks...")
    chunks = [
        "Claude is an AI assistant created by Anthropic, designed to be helpful, harmless, and honest.",
        "Anthropic is an AI safety company focused on developing safe and beneficial AI systems.",
        "The Transformer architecture revolutionized natural language processing and is used in many modern AI systems.",
        "Knowledge graphs store structured information about entities and their relationships.",
        "The Model Context Protocol (MCP) enables secure communication between AI models and external tools."
    ]
    
    for chunk in chunks:
        store_chunk(chunk)
        time.sleep(0.1)
    
    # Verify data was loaded
    print("\n4. Verifying data...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/metrics")
        if response.status_code == 200:
            data = response.json()
            stats = data.get("data", {})
            print(f"   Entities: {stats.get('entity_count', 0)}")
            print(f"   Memory stats: {stats.get('memory_stats', {})}")
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/query",
            headers={"Content-Type": "application/json"},
            json={"limit": 50}
        )
        
        if response.status_code == 200:
            data = response.json()
            triples = data.get("data", {}).get("triples", [])
            print(f"   Triples: {len(triples)}")
            if triples:
                print("   Sample triples:")
                for triple in triples[:3]:
                    print(f"     - {triple.get('subject')} {triple.get('predicate')} {triple.get('object')}")
    except Exception as e:
        print(f"   Error verifying: {e}")
    
    print("\nSUCCESS: Data reload complete!")
    print("Visit http://localhost:8090 to see the knowledge graph visualization")

if __name__ == "__main__":
    main()