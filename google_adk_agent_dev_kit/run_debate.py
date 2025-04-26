"""
Main script to run the Earth shape debate using ADK agents.
"""

import os
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from agents import create_agents
from mock_data import EVIDENCE_DATA, HISTORICAL_REFERENCES, IMAGE_REFERENCES

def check_tools():
    """Test the tools to ensure they're working properly."""
    print("\nNote: Tool testing is limited because we can't fully simulate the ToolContext.")
    print("In a real debate, the tools will infer the perspective from the calling agent.\n")

    from tools import evidence_retriever, historical_reference, image_reference
    from google.adk.tools.tool_context import ToolContext

    # Create a basic tool context
    # We can't properly test agent perspective inference without the actual agents
    tool_context = ToolContext(state={})

    # Test both evidence perspectives manually
    print("Testing evidence for Round Earth perspective:")
    round_result = EVIDENCE_DATA["horizon"]["round_earth"]
    print(f"  Result: {round_result}\n")

    print("Testing evidence for Flat Earth perspective:")
    flat_result = EVIDENCE_DATA["horizon"]["flat_earth"]
    print(f"  Result: {flat_result}\n")

    # Test basic evidence_retriever tool function:
    print("Testing basic evidence_retriever tool function:")
    # Note: This call won't have agent perspective, so it will return both
    basic_result = evidence_retriever("horizon", tool_context)
    print(f"  Result: {basic_result}\n")

    print("Tool tests completed. Note that live perspective will be detected during the debate.")

def run_debate():
    """Run the Earth shape debate with all three agents."""

    import time
    import random

    # Helper function for rate limiting
    def wait_with_jitter(base_seconds=3):
        """Wait with jitter to avoid rate limiting"""
        jitter = random.uniform(1, 3)  # Add 1-3 seconds of jitter
        wait_time = base_seconds + jitter
        print(f"[Waiting for model response...]")
        time.sleep(wait_time)

    # Uncomment this to run tool checks
    # check_tools()
    print("\n[Initializing debate simulation...]\n")
    # Create the agents
    try:
        # model_to_use = "gemini-2.5-flash-preview-04-17"
        model_to_use = "anthropic/claude-3-7-sonnet-latest"
        moderator_agent, round_earth_agent, flat_earth_agent = create_agents( model_to_use )
    except Exception as e:
        print(f"ERROR: Failed to create agents: {e}")
        return

    # Create a session service to maintain state
    session_service = InMemorySessionService()

    # Create a session for the debate
    session_id = "debate_session"
    user_id = "debate_user"
    session = session_service.create_session(
        app_name="earth_shape_debate",
        user_id=user_id,
        session_id=session_id
    )

    # Create a runner for the moderator agent
    runner = Runner(
        agent=moderator_agent,
        app_name="earth_shape_debate",
        session_service=session_service
    )

    print("=" * 80)
    print("EARTH SHAPE DEBATE")
    print("=" * 80)
    print("\nWelcome to the Earth shape debate!")
    print("The Moderator will guide a discussion between a Round Earth Advocate and a Flat Earth Advocate.")
    print("The debate will cover several topics with each side presenting their arguments.")
    print("\nTo start the debate, press Enter.")

    # print("[DEBUG] Before first input()") # DEBUG PRINT 1
    input()
    # print("[DEBUG] After first input(), before initial runner.run") # DEBUG PRINT 2

    # Start the debate with an initial prompt
    query = """Please introduce the debate and the participants. Then, get opening statements from BOTH advocates - first from the round_earth_advocate and then from the flat_earth_advocate. Make sure both advocates have a chance to speak before moving on."""
    # Add reminder comments to the prompt that both advocates must speak

    # Simulate a multi-turn conversation for the debate
    debate_topics = [
        "Now, let's discuss visual evidence. First, ask the round_earth_advocate what we can observe about the horizon and from space. After they respond, ask the flat_earth_advocate for their perspective on the same topic.",
        "Let's move on to physical evidence. First, ask the round_earth_advocate about gravity and circumnavigation experiences. Then, ask the flat_earth_advocate for their perspective on these physical phenomena.",
        "What about historical perspectives on Earth's shape? First, get the round_earth_advocate's historical view, then ask the flat_earth_advocate about historical perspectives supporting their position.",
        "Thank you both. Please first ask the round_earth_advocate for their closing statement summarizing key points, then ask the flat_earth_advocate for their closing statement."
    ]

    # First query to start the debate
    print("\nPrompt: " + query)
    print("\n" + "=" * 80 + "\n")

    # Process the query
    try:
        # Initialize session state for tracking the debate
        session.state["current_topic"] = "opening_statements"

        content = types.Content(role='user', parts=[types.Part(text=query)])
        response_events = runner.run(user_id=user_id, session_id=session_id, new_message=content)
        for event in response_events:
            # Print speaker and content if available
            if hasattr(event, 'content') and event.content and event.content.parts:
                speaker = event.author

                # Print speaker header with formatting
                print(f"\n{'-' * 20}")
                print(f"SPEAKER: {speaker}")
                print(f"{'-' * 20}\n")

                # Print content
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        print(part.text)
                    # Silently skip non-text parts without printing warnings
    except Exception as e:
        print(f"ERROR: Failed to process query: {e}")
        return

    # print("[DEBUG] After initial runner.run and event processing, before topic loop") # DEBUG PRINT 3

    # Go through each debate topic
    for topic in debate_topics:
        # print("[DEBUG] Inside topic loop, before 'Press Enter' prompt") # DEBUG PRINT 4
        print("\n" + "=" * 80)
        print("\nPress Enter to continue to the next topic...")
        input()
        # print("[DEBUG] After 'Press Enter' prompt, before topic runner.run") # DEBUG PRINT 5

        print("\nPrompt: " + topic)
        print("\n" + "=" * 80 + "\n")

        # Process the query
        try:
            content = types.Content(role='user', parts=[types.Part(text=topic)])

            # Add rate limiting to avoid API quotas
            wait_with_jitter(5)  # Wait longer for the first response

            response_events = runner.run(user_id=user_id, session_id=session_id, new_message=content)

            # Track which advocates have spoken (for backup system)
            round_earth_spoke = False
            flat_earth_spoke = False

            for event in response_events:
                # Print speaker and content if available
                if hasattr(event, 'content') and event.content and event.content.parts:
                    speaker = event.author

                    # Track which advocates have spoken
                    if speaker == "round_earth_advocate":
                        round_earth_spoke = True
                    elif speaker == "flat_earth_advocate":
                        flat_earth_spoke = True

                    # Print speaker header with formatting
                    print(f"\n{'-' * 20}")
                    print(f"SPEAKER: {speaker}")
                    print(f"{'-' * 20}\n")

                    # Print content
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            print(part.text)
                        # Silently skip non-text parts without printing warnings

            # If only one spoke, try to prompt for the other (without debug printing)
            if round_earth_spoke and not flat_earth_spoke:
                print("\n[Moderator asks the Flat Earth Advocate to respond...]\n")
                try:
                    # Add delay to avoid rate limiting
                    wait_with_jitter(3)

                    # Create a direct prompt for the flat earth advocate
                    followup = f"""IMPORTANT: We need to hear from the flat_earth_advocate who has not spoken yet.
                    Moderator, please do the following:
                    1. Explicitly call the flat_earth_advocate by saying 'flat_earth_advocate, please present your perspective'
                    2. Give them the floor to respond to this topic: {topic}
                    3. Make sure to wait for their complete response

                    The flat_earth_advocate MUST speak now."""
                    # Skip printing the special prompt to keep output clean
                    content = types.Content(role='user', parts=[types.Part(text=followup)])
                    followup_events = runner.run(user_id=user_id, session_id=session_id, new_message=content)

                    for event in followup_events:
                        if hasattr(event, 'content') and event.content and event.content.parts:
                            speaker = event.author
                            print(f"\n{'-' * 20}")
                            print(f"SPEAKER: {speaker}")
                            print(f"{'-' * 20}\n")

                            for part in event.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    print(part.text)
                                # Silently skip non-text parts without printing warnings
                except Exception as e:
                    print(f"ERROR: Failed to prompt Flat Earth Advocate: {e}")
                    pass # Continue even if followup fails
        except Exception as e:
            print(f"ERROR: Failed to process topic: {e}")
            continue # Continue to the next topic if this one fails

    print("\n" + "=" * 80)
    print("\nThe debate has concluded. Thank you for attending!")
    print("=" * 80)

if __name__ == "__main__":
    try:
        run_debate()
    except Exception as e:
        import traceback
        print("\n" + "=" * 80)
        print(f"ERROR: Debate failed with exception: {e}")
        print("Traceback:")
        traceback.print_exc()
        print("=" * 80)
