# rate limits
Warning: on the free tier, this program often exceeds the minute limit for gemini 2.5 flash.  You may need to increase waits in the code.

# Earth Shape Debate Agents

A demonstration of Google's Agent Development Kit (ADK) featuring a debate between agents on the shape of the Earth, moderated by a third agent.

## Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

1. Create and activate a virtual environment:
```bash
# Create the virtual environment
python -m venv .venv

# Activate it (use the appropriate command for your OS)
# macOS/Linux:
source .venv/bin/activate
# Windows (CMD):
.venv\Scripts\activate.bat
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the debate:
```bash
python run_debate.py
```

## System Architecture

This demo consists of three agents:

1. **Moderator Agent**: Controls the debate flow, manages turns, and provides summaries
2. **Round Earth Advocate**: Presents evidence supporting a spherical Earth
3. **Flat Earth Advocate**: Presents arguments supporting a flat Earth model

The agents share common tools for retrieving evidence and references, demonstrating how the same tools can be used by different agents with different perspectives.

### Key Components:

- `tools.py`: Defines shared tools for retrieving evidence and references
- `agents.py`: Defines the three debate agents with different roles and instructions
- `run_debate.py`: Main script to run the debate
- `mock_data.py`: Contains mock data for the tools to use

### ADK Features Demonstrated:

1. **Multi-agent orchestration**: How a coordinator agent (Moderator) can manage the flow between specialized agents
2. **Tool sharing**: How multiple agents can use the same tools but with different interpretations
3. **State management**: How agents maintain context throughout the conversation
4. **Agent delegation**: How the Moderator delegates to specialized agents at appropriate times

## Extending the Demo

You can extend this demo by:
- Adding more specialized tools
- Creating additional debate agents with different perspectives
- Implementing more complex state management
- Adding additional debate topics

## License

This project is provided for educational purposes.
