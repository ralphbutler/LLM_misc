# rate limits
Warning: on the free tier, this program often exceeds the minute limit for gemini 2.5 flash.  You may need to increase waits in the code.

# Earth Shape Debate Agents

Great! I've implemented a complete debate system using Google's Agent Development Kit. Here's an overview of what I've created:

1. **Core Components**:
   - Three specialized agents: Moderator, Round Earth Advocate, and Flat Earth Advocate
   - Three shared tools for both debaters: `evidence_retriever`, `historical_reference`, and `image_reference`
   - Mock data for the tools to use

2. **Key Files**:
   - `agents.py`: Defines the three debate agents with their specific instructions
   - `tools.py`: Implements the shared tools for retrieving evidence and references
   - `mock_data.py`: Contains the mock data for the tools
   - `run_debate.py`: Main script to run the debate
   - `config.py`: Handles environment configuration
   - Supporting files for setup and documentation

3. **ADK Features Demonstrated**:
   - **Multi-agent Orchestration**: The Moderator agent coordinates the debate flow and delegates to specialized advocate agents
   - **Tool Sharing**: Both advocate agents access the same tools but use them differently based on their perspectives
   - **State Management**: Tools track their usage in the session state, and the Moderator saves the current topic
   - **Agent Delegation**: The Moderator delegates to specific agents based on the debate structure
   - **Agent Specialization**: Each agent has distinct roles, instructions, and ways of interpreting the same data

4. **How To Run**:
   - Install dependencies with `pip install -r requirements.txt`
   - Set up your API credentials in a `.env` file (based on `.env.example`)
   - Run the debate with `python run_debate.py`
   - The script will guide the user through the debate with prompts between topics

5. **Technical Implementation Details**:
   - Uses ADK's `Agent` class to create the three specialized agents
   - Uses the `InMemorySessionService` for simple state management
   - Implements tools with the `ToolContext` parameter to access session state
   - Uses the `Runner` class to manage the conversation flow
   - The mock data structure allows both perspectives to reference the same topics

This system offers a compact demonstration of the Google ADK's core features in an engaging and educational format. The debate progresses through structured topics with each agent playing its defined role, showing how multi-agent systems can be orchestrated to create interactive experiences.
