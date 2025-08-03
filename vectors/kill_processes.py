#!/usr/bin/env python3
"""
Process Killer for LLMKG Vector System
Stops any runaway Python processes consuming CPU
"""

import psutil
import sys
import os
from pathlib import Path

def find_vector_processes():
    """Find Python processes running vector-related scripts."""
    vector_processes = []
    vector_script_names = [
        'indexer_bge_optimized.py',
        'indexer_i9_optimized.py', 
        'indexer_hd.py',
        'indexer_mpnet.py',
        'query_bge.py',
        'query_hd.py',
        'query.py',
        'cross_reference_validator.py',
        'validate_bge_large.py',
        'test_bge_database.py'
    ]
    
    current_pid = os.getpid()  # Don't kill ourselves
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent']):
        try:
            if proc.info['pid'] == current_pid:
                continue
                
            if proc.info['name'] == 'python.exe' or proc.info['name'] == 'python':
                cmdline = proc.info['cmdline']
                if cmdline and len(cmdline) > 1:
                    script_name = Path(cmdline[1]).name if len(cmdline) > 1 else ""
                    
                    if script_name in vector_script_names:
                        cpu_usage = proc.cpu_percent(interval=1)  # Get CPU usage
                        vector_processes.append({
                            'pid': proc.info['pid'],
                            'script': script_name,
                            'cpu_percent': cpu_usage,
                            'cmdline': ' '.join(cmdline)
                        })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return vector_processes

def kill_high_cpu_processes(processes, cpu_threshold=50.0):
    """Kill processes using high CPU."""
    killed = []
    
    for proc_info in processes:
        if proc_info['cpu_percent'] > cpu_threshold:
            try:
                proc = psutil.Process(proc_info['pid'])
                print(f"Killing high-CPU process: PID {proc_info['pid']} ({proc_info['script']}) - {proc_info['cpu_percent']:.1f}% CPU")
                
                proc.terminate()  # Try graceful termination first
                try:
                    proc.wait(timeout=5)  # Wait up to 5 seconds
                except psutil.TimeoutExpired:
                    print(f"  Force killing PID {proc_info['pid']}")
                    proc.kill()  # Force kill if needed
                
                killed.append(proc_info)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"  Could not kill PID {proc_info['pid']}: {e}")
    
    return killed

def main():
    """Main process killer function."""
    print("=" * 60)
    print("LLMKG VECTOR PROCESS KILLER")
    print("=" * 60)
    print("Searching for Python vector processes...")
    
    # Find vector-related processes
    processes = find_vector_processes()
    
    if not processes:
        print("[OK] No vector-related Python processes found")
        return
    
    print(f"\nFound {len(processes)} vector-related processes:")
    print("-" * 60)
    
    for proc in processes:
        print(f"PID {proc['pid']:>8} | {proc['cpu_percent']:>6.1f}% CPU | {proc['script']}")
    
    print("-" * 60)
    
    # Check for high CPU usage
    high_cpu_procs = [p for p in processes if p['cpu_percent'] > 10.0]
    
    if high_cpu_procs:
        print(f"\nFound {len(high_cpu_procs)} high-CPU processes (>10% CPU)")
        
        choice = input("\nKill high-CPU processes? [y/N]: ").strip().lower()
        
        if choice in ['y', 'yes']:
            killed = kill_high_cpu_processes(high_cpu_procs, cpu_threshold=10.0)
            print(f"\nKilled {len(killed)} processes")
            
            if killed:
                print("Killed processes:")
                for proc in killed:
                    print(f"  - PID {proc['pid']} ({proc['script']})")
        else:
            print("No processes killed")
    else:
        print("\n[OK] No high-CPU processes found")
        
        if processes:
            choice = input("\nKill all vector processes anyway? [y/N]: ").strip().lower()
            if choice in ['y', 'yes']:
                killed = kill_high_cpu_processes(processes, cpu_threshold=0.0)
                print(f"\nKilled {len(killed)} processes")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)