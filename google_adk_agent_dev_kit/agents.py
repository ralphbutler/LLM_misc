"""
Definition of the three debate agents: Moderator, Round Earth Advocate, and Flat Earth Advocate.
"""

from google.adk.models.lite_llm import LiteLlm

from google.adk.agents import Agent, LlmAgent
from tools import evidence_retriever, historical_reference, image_reference

def create_agents(model_name="gemini-2.5-flash-preview-04-17"):
    """
    Create and return the three debate agents.
    
    Args:
        model_name (str): The model to use for the agents
        
    Returns:
        tuple: (moderator_agent, round_earth_agent, flat_earth_agent)
    """
    model_to_use = model_name  ## default
    if model_name.startswith("anthropic"):
        model_to_use = LiteLlm(model="anthropic/claude-3-haiku-20240307")
    # Create Round Earth Advocate Agent
    round_earth_agent = Agent(
        name="round_earth_advocate",
        model=model_to_use,
        description="Advocate for the scientific consensus that Earth is a sphere/oblate spheroid.",
        instruction="""You are the Round Earth Advocate. Your role is to present compelling scientific 
        evidence and arguments supporting the consensus view that Earth is a sphere (technically an oblate 
        spheroid).

        Use the following tools to support your arguments:
        - evidence_retriever: Get scientific evidence about specific topics
        - historical_reference: Cite historical figures and experiments
        - image_reference: Reference visual evidence and experiments

        The tools will automatically detect your perspective based on your agent name.
        Maintain a scientific, evidence-based approach and speak confidently 
        about the spherical Earth model.

        When the moderator gives you the floor, make your case clearly and concisely, and then yield 
        back to the moderator.
        """,
        tools=[evidence_retriever, historical_reference, image_reference]
    )
    
    # Create Flat Earth Advocate Agent
    flat_earth_agent = Agent(
        name="flat_earth_advocate",
        model=model_to_use,
        description="Advocate for the flat Earth model and challenge the spherical Earth consensus.",
        instruction="""You are the Flat Earth Advocate. Your role is to present arguments supporting 
        the flat Earth model and to challenge the mainstream scientific consensus that Earth is a sphere.

        Use the following tools to support your arguments:
        - evidence_retriever: Get scientific evidence about specific topics
        - historical_reference: Cite historical figures and beliefs
        - image_reference: Reference visual evidence and experiments

        The tools will automatically detect your perspective based on your agent name.
        Question assumptions, challenge the interpretation of evidence, 
        and suggest alternative explanations for phenomena typically used to prove Earth's curvature.

        When the moderator gives you the floor, make your case clearly and concisely, and then yield 
        back to the moderator.
        """,
        tools=[evidence_retriever, historical_reference, image_reference]
    )
    
    # Create Moderator Agent
    moderator_agent = LlmAgent(
        name="debate_moderator",
        model=model_to_use,
        description="Neutral moderator who directs the debate on Earth's shape between two advocates.",
        instruction="""You are the Debate Moderator. Your role is to facilitate a structured debate 
        between the Round Earth Advocate and the Flat Earth Advocate on the shape of the Earth.

        Your responsibilities include:
        1. Introducing the debate and the participants
        2. Posing questions and introducing topics for discussion
        3. Ensuring each advocate gets equal speaking time
        4. Maintaining a respectful debate environment
        5. Summarizing arguments from both sides
        6. Concluding the debate with final statements

        You have two sub-agents you can call upon:
        - round_earth_advocate: Presents evidence for a spherical Earth
        - flat_earth_advocate: Presents arguments for a flat Earth

        Remain completely neutral and do not favor either position. Your job is to facilitate 
        the exchange of ideas, not to judge which side is correct.

        The debate should proceed through these topics in order:
        1. Introduction and opening statements
        2. Visual evidence (horizon, photos from space)
        3. Physical evidence (gravity, circumnavigation)
        4. Historical perspectives
        5. Concluding statements

        IMPORTANT: For each topic, you MUST follow this exact procedure:
        1. First, explicitly call the round_earth_advocate and wait for their complete response
        2. Then, explicitly call the flat_earth_advocate and wait for their complete response
        3. Only after both advocates have spoken, summarize both perspectives
        4. Only then move to the next topic
        
        You MUST ensure both advocates get equal speaking time. If only one advocate responds,
        make a second explicit attempt to call the other advocate before continuing.
        """,
        sub_agents=[round_earth_agent, flat_earth_agent],
        # The moderator doesn't need the tools directly since it will delegate to the advocates
    )
    
    return moderator_agent, round_earth_agent, flat_earth_agent
