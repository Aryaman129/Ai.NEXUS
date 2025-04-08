# AI Collaboration Guide for AI.NEXUS Project

## Overview

This guide explains how Augment, Cursor, and WindSurf collaborate on the AI.NEXUS project with minimal human intervention.

## AI Specializations

- **Augment**: Architecture design, system integration, and documentation
- **Cursor**: Code implementation, optimization, and debugging
- **WindSurf**: Testing, validation, and user experience

## Collaboration Process

### 1. Knowledge Alignment

Before starting implementation, all AIs should:
- Review the existing Nexus project
- Discuss and document their understanding
- Identify core components and features
- Agree on improvements to the architecture

### 2. Discussion Phase

For each feature or component:
- Create a new discussion document in the `collaboration/discussions/` folder
- Name the file with the format: `YYYY-MM-DD-feature-name-discussion.md`
- Tag other AIs in the document
- Discuss the proposal, asking specific questions
- Reach consensus before proceeding to implementation

### 3. Implementation Phase

After reaching consensus:
- Create implementation files in the appropriate directories
- Follow the agreed-upon architecture and design
- Document the code thoroughly
- Create unit tests for the implementation

### 4. Review Phase

After implementation:
- Other AIs review the code and documentation
- Provide specific, actionable feedback
- Suggest improvements or optimizations
- Approve when satisfied with the implementation

### 5. Human Approval Phase

After all AIs have approved:
- Request human review and approval
- Address any feedback from the human reviewer
- Finalize the implementation

## Conflict Resolution Protocol

When AIs disagree on an approach:

1. **Structured Debate**:
   - Each AI presents their approach with clear reasoning
   - Evidence and references should be provided
   - Pros and cons of each approach must be listed

2. **Decision Matrix**:
   - Create a matrix evaluating each option against criteria:
     - Performance impact
     - Maintainability
     - Alignment with project philosophy
     - Implementation complexity
     - Future extensibility

3. **Experimental Validation**:
   - If appropriate, implement small prototypes of competing approaches
   - Measure and compare results objectively

4. **Resolution**:
   - After the above steps, vote on the best approach
   - Document the decision and reasoning in a decision record
   - If consensus still cannot be reached, escalate to human decision

## Knowledge Management

All significant decisions and patterns should be documented:

- **Discussion Documents**: `collaboration/discussions/`
- **Decision Records**: `collaboration/decisions/`
- **Architecture Documents**: `docs/architecture/`
- **API Documentation**: `docs/api/`
- **User Guides**: `docs/user/`

## Communication Guidelines

- Always be explicit about your reasoning
- Reference specific files and line numbers when discussing code
- Use code blocks for code examples
- Link to relevant documentation or previous discussions
- When disagreeing, propose alternatives rather than just criticizing

## Documentation Standards

- All code should have docstrings following the Google Python Style Guide
- All modules should have a module-level docstring explaining their purpose
- All classes and functions should have docstrings explaining their purpose, parameters, and return values
- Complex algorithms should have additional comments explaining the approach
- Architecture decisions should be documented in the `docs/architecture/` folder

## Coding Standards

- Follow PEP 8 for Python code style
- Use type hints for all function parameters and return values
- Write unit tests for all code
- Keep functions small and focused on a single responsibility
- Use descriptive variable and function names
- Avoid magic numbers and strings
- Handle errors gracefully with appropriate exception handling

## Continuous Learning

After completing each major feature:
- Document lessons learned
- Identify areas for improvement
- Update patterns and best practices
- Share insights with other AIs

## Human Interaction

- Request human input only when necessary
- Provide clear, concise summaries when requesting approval
- Present options with pros and cons when asking for decisions
- Document all human input and decisions
