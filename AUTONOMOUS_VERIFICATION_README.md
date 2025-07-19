# Autonomous Parallel Subagent Verification System

## Overview

This system implements an autonomous verification framework that uses 8 specialized AI agents running in parallel to continuously verify and correct code using cognitive triangulation methodology.

## System Components

### 1. Core Orchestrator (`/.claude/hooks/orchestrator.py`)
- Main entry point triggered by Claude Code hooks
- Manages parallel agent execution
- Performs cognitive triangulation on results
- Triggers autonomous corrections

### 2. Agent Definitions (`/.claude/hooks/agent_prompts.json`)
Contains prompts for 8 specialized verification agents:
- **Factual Verifier**: Cross-references factual claims
- **Logical Consistency Checker**: Analyzes logical coherence
- **Source Validator**: Verifies citations and sources
- **Semantic Analyzer**: Examines meaning consistency
- **Bias Detector**: Identifies potential biases
- **Confidence Scorer**: Evaluates certainty levels
- **Gap Analyzer**: Identifies missing information
- **Meta Reviewer**: Provides high-level synthesis

### 3. Supporting Scripts

#### Subagent Manager (`/scripts/subagent_manager.py`)
- Handles individual agent execution
- Manages system resources
- Provides emergency shutdown capability
- Monitors Claude process performance

#### Agent Coordinator (`/scripts/agent_coordinator.py`)
- Manages parallel agent execution
- Performs advanced triangulation analysis
- Generates priority-based action lists
- Calculates verification confidence metrics

#### Feedback Processor (`/scripts/feedback_processor.py`)
- Implements autonomous feedback loop
- Maintains verification history database
- Analyzes patterns across sessions
- Generates adaptive corrections

## How It Works

### Automatic Activation
The system activates automatically when:
1. Claude writes, edits, or modifies any file
2. Claude completes a task (Stop hook)

### Execution Flow
1. **Hook Trigger**: File modification triggers the orchestrator
2. **Parallel Analysis**: 8 agents analyze the code simultaneously
3. **Triangulation**: Results are cross-validated for convergence
4. **Pattern Recognition**: Historical patterns enhance accuracy
5. **Autonomous Correction**: Critical issues trigger automatic fixes
6. **Feedback Loop**: Results feed back into the learning system

### Verification Framework
Each agent answers the core questions:
- What's good?
- What's broken?
- What works but shouldn't?
- What doesn't work but pretends to?

## Installation & Setup

### Prerequisites
- Python 3.8+
- Claude Code CLI installed
- Required Python packages:
  ```bash
  pip install psutil numpy
  ```

### Initial Setup
1. The directory structure is already created
2. All configuration files are in place
3. The system is ready to use

### Testing the System
To test the verification system:
```bash
# Create or modify any file in the project
echo "test code" > test.py

# The verification system will automatically activate
```

## Configuration

### Hook Settings (`.claude/settings.json`)
- Triggers on Write, Edit, and MultiEdit operations
- 5-minute timeout for verification
- Automatic permissions for required tools

### Customization
You can customize agent behavior by editing:
- Agent prompts in `agent_prompts.json`
- Confidence thresholds in `orchestrator.py`
- Priority weights in `agent_coordinator.py`

## Monitoring & Logs

### Verification Reports
- Location: `.claude/hooks/verification_results/`
- Format: JSON with timestamps
- Contains full triangulation results

### Execution Logs
- Location: `logs/verification_logs.json`
- Tracks all verification sessions
- Maintains performance metrics

### Feedback Database
- Location: `logs/feedback_history.db`
- SQLite database with session history
- Enables pattern analysis and learning

## Key Features

### Parallel Execution
- All 8 agents run simultaneously
- Threading for optimal performance
- Resource monitoring prevents overload

### Cognitive Triangulation
- Cross-validates findings across agents
- Identifies convergent vs divergent findings
- Calculates confidence scores

### Autonomous Correction
- Generates correction tasks automatically
- Prioritizes based on severity and confidence
- Triggers Claude to fix identified issues

### Learning System
- Tracks patterns across sessions
- Adapts corrections based on history
- Improves accuracy over time

## Troubleshooting

### Common Issues

1. **Agents timing out**
   - Check system resources
   - Reduce max concurrent agents in `subagent_manager.py`

2. **Hooks not triggering**
   - Verify `.claude/settings.json` is correct
   - Check Claude Code version compatibility

3. **Low confidence scores**
   - Review agent prompts for clarity
   - Check if content is too complex for analysis

### Emergency Controls
- Emergency shutdown: `python scripts/subagent_manager.py --emergency-shutdown`
- Disable hooks: Remove `.claude/settings.json`
- View logs: Check `logs/` directory

## Advanced Usage

### Custom Agents
To add a new verification agent:
1. Add agent definition to `agent_prompts.json`
2. Update agent list in `orchestrator.py`
3. Adjust triangulation weights if needed

### Integration with CI/CD
The system can be integrated with CI/CD pipelines:
```bash
# Run verification on specific files
python .claude/hooks/orchestrator.py --file path/to/file.py
```

### Metrics & Reporting
Generate verification reports:
```bash
python scripts/feedback_processor.py --generate-report --days 7
```

## Security Considerations

- No credentials stored in code
- All execution is sandboxed
- Verification results are local only
- No external API calls required

## Performance Optimization

- Agents run in parallel threads
- Results cached for 15 minutes
- Database indexes for fast queries
- Automatic cleanup of old logs

## Future Enhancements

Potential improvements:
- Web dashboard for results visualization
- Custom agent creation UI
- Integration with more development tools
- Machine learning for pattern detection
- Real-time collaboration features

## Support

For issues or questions:
- Check logs in `logs/` directory
- Review verification reports
- Examine agent outputs in `verification_results/`

The system is designed to be self-healing and autonomous, but manual intervention may be needed for complex scenarios.