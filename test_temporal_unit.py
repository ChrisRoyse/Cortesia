#!/usr/bin/env python3
"""Unit test for temporal tracking functionality."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.mcp.llm_friendly_server.temporal_tracking import TemporalIndex, TemporalOperation
from src.core.triple import Triple
from datetime import datetime
import time

def test_temporal_tracking():
    """Test the temporal tracking functionality directly."""
    
    # Create a temporal index
    index = TemporalIndex()
    
    # Test 1: Record a CREATE operation
    print("1. Testing CREATE operation...")
    triple1 = Triple(
        subject="Einstein",
        predicate="occupation",
        object="physicist",
        confidence=1.0,
        source=None
    )
    index.record_operation(triple1, TemporalOperation.Create, None)
    print("✓ CREATE operation recorded")
    
    # Small delay to ensure different timestamps
    time.sleep(0.1)
    
    # Test 2: Record an UPDATE operation
    print("\n2. Testing UPDATE operation...")
    triple2 = Triple(
        subject="Einstein",
        predicate="occupation",
        object="theoretical physicist",
        confidence=1.0,
        source=None
    )
    index.record_operation(triple2, TemporalOperation.Update, "physicist")
    print("✓ UPDATE operation recorded with previous value")
    
    # Test 3: Check entity timeline
    print("\n3. Checking entity timeline...")
    timelines = index.entity_timelines.read().unwrap()
    
    if "Einstein" in timelines:
        einstein_timeline = timelines["Einstein"]
        print(f"✓ Found timeline for Einstein with {len(einstein_timeline)} timestamps")
        
        for ts, changes in einstein_timeline.items():
            print(f"\n  Timestamp: {ts}")
            for change in changes:
                print(f"    - Operation: {change.operation.name}")
                print(f"      Triple: {change.triple.subject} {change.triple.predicate} {change.triple.object}")
                if change.previous_value:
                    print(f"      Previous value: {change.previous_value}")
    else:
        print("✗ No timeline found for Einstein")
    
    # Test 4: Check global timeline
    print("\n4. Checking global timeline...")
    global_timeline = index.global_timeline.read().unwrap()
    print(f"✓ Global timeline has {len(global_timeline)} timestamps")
    
    total_operations = sum(len(changes) for changes in global_timeline.values())
    print(f"✓ Total operations recorded: {total_operations}")
    
    # Test 5: Test version tracking
    print("\n5. Testing version tracking...")
    triple3 = Triple(
        subject="Einstein",
        predicate="born_in",
        object="1879",
        confidence=1.0,
        source=None
    )
    index.record_operation(triple3, TemporalOperation.Create, None)
    
    # Check version numbers
    timelines = index.entity_timelines.read().unwrap()
    einstein_timeline = timelines["Einstein"]
    
    versions = []
    for ts, changes in einstein_timeline.items():
        for change in changes:
            versions.append(change.version)
    
    print(f"✓ Version numbers: {versions}")
    assert versions == sorted(versions), "Versions should be increasing"
    
    print("\n✅ All temporal tracking tests passed!")

if __name__ == "__main__":
    print("Temporal Tracking Unit Test")
    print("===========================")
    test_temporal_tracking()