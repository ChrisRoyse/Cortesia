# Multi-Agent Orchestration & Validation Strategy

This document outlines the orchestration and validation strategy for the 5-agent swarm system. The parent agent is the central conductor, responsible for managing the lifecycle of all subagents and ensuring the integrity of all operations through a rigorous validation protocol.

## Parent Orchestrator Logic

For any complex task, the parent orchestrator must:

1.  **Spawn 5 Specialized Agents**: At the start of any task, the parent must spawn the 5 core agents, each assigned a specific sub-task as defined in the `agent_definitions.json` file.
2.  **Monitor Subagent Completion**: The parent must continuously monitor the `.claude/sub_agents/` directory for completion reports from each subagent.
3.  **Enforce Persistence**: If a subagent completes its task and the overall project objectives are not yet met, the parent must immediately re-spawn that agent with its original instructions. Subagent slots must never be left empty.
4.  **Trigger Validation on Write**: On every file read or write action, the parent agent must trigger the 5-agent validation swarm to independently validate the operation.
5.  **Aggregate Validation Results**: The parent agent must aggregate the validation reports from all 5 subagents and halt the workflow if validation fails. The parent may only proceed on a majority endorsement from the validation swarm.

## Subagent Template

Each of the 5 core agents must be spawned with the following template:

-   **Goal**: [Describe the sub-task from the `agent_definitions.json` file]
-   **Output**: Save all work and a completion report to `.claude/sub_agents/[AGENT_NAME]_report.md`.
-   **Signal Completion**: Upon completion of a discrete task, signal the parent orchestrator by writing the completion report.

## Validation Protocol

-   **Trigger**: Any `Write`, `Edit`, or `MultiEdit` action by the parent agent.
-   **Action**: The `validation_orchestrator.py` script is triggered, which spawns the 5 core agents.
-   **Task**: Each agent independently validates the parent's action against the project's to-do list and overall objectives.
-   **Output**: Each validation agent writes a report to `.claude/sub_agents/[AGENT_NAME]_validation.md`.
-   **Consensus**: The parent agent requires a majority of positive validations to proceed.

This `CLAUDE.md` file serves as the primary source of truth for the orchestration and validation logic. All agents, especially the parent orchestrator, must adhere to these principles to ensure a robust, persistent, and successful multi-agent workflow.