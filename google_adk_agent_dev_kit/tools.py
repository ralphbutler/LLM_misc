"""
Shared tools for both debate agents to use.
"""

from mock_data import EVIDENCE_DATA, HISTORICAL_REFERENCES, IMAGE_REFERENCES
from google.adk.tools.tool_context import ToolContext

def evidence_retriever(topic: str, tool_context: ToolContext = None) -> dict:
    """
    Retrieves evidence on a specific Earth shape topic.
    
    Args:
        topic (str): The topic to retrieve evidence for (e.g., "horizon", "gravity")
        tool_context (ToolContext, optional): The tool context, used to infer perspective
    
    Returns:
        dict: Evidence data including facts and explanations
    """
    perspective = ""
    # Track usage in state if context is provided
    if tool_context:
        topics_queried = tool_context.state.get("topics_queried", [])
        if topic not in topics_queried:
            topics_queried.append(topic)
            tool_context.state["topics_queried"] = topics_queried
        
        # Try to infer perspective from agent name in tool_context
        # In the ADK, we can infer this from the calling agent via tool_context
        if tool_context:
            calling_agent = getattr(tool_context, 'calling_agent', None) or ''
            if "flat" in calling_agent.lower():
                perspective = "flat_earth"
            elif "round" in calling_agent.lower():
                perspective = "round_earth"
    
    # Normalize topic input
    normalized_topic = topic.lower().strip()
    
    # Check if topic exists
    if normalized_topic not in EVIDENCE_DATA:
        return {
            "status": "error",
            "error_message": f"No evidence found for topic: {topic}. Available topics: {', '.join(EVIDENCE_DATA.keys())}"
        }
    
    # Return evidence based on perspective
    if perspective == "round_earth":
        return {
            "status": "success",
            "evidence": EVIDENCE_DATA[normalized_topic]["round_earth"],
            "topic": normalized_topic
        }
    elif perspective == "flat_earth":
        return {
            "status": "success",
            "evidence": EVIDENCE_DATA[normalized_topic]["flat_earth"],
            "topic": normalized_topic
        }
    else:
        # Return both perspectives
        return {
            "status": "success",
            "evidence": {
                "round_earth": EVIDENCE_DATA[normalized_topic]["round_earth"],
                "flat_earth": EVIDENCE_DATA[normalized_topic]["flat_earth"]
            },
            "topic": normalized_topic
        }

def historical_reference(period: str, tool_context: ToolContext = None) -> dict:
    """
    Retrieves historical references about Earth shape theories.
    
    Args:
        period (str): Historical period (e.g., "ancient_greece", "medieval_period")
        tool_context (ToolContext, optional): The tool context for state management
    
    Returns:
        dict: Historical reference data
    """
    perspective = ""
    # Track usage in state if context is provided
    if tool_context:
        periods_queried = tool_context.state.get("periods_queried", [])
        if period not in periods_queried:
            periods_queried.append(period)
            tool_context.state["periods_queried"] = periods_queried
        
        # Try to infer perspective from agent name in tool_context
        # In the ADK, we can infer this from the calling agent via tool_context
        if tool_context:
            calling_agent = getattr(tool_context, 'calling_agent', None) or ''
            if "flat" in calling_agent.lower():
                perspective = "flat_earth"
            elif "round" in calling_agent.lower():
                perspective = "round_earth"
    
    # Normalize period input
    normalized_period = period.lower().strip()
    
    # Check if period exists
    if normalized_period not in HISTORICAL_REFERENCES:
        return {
            "status": "error",
            "error_message": f"No historical references found for period: {period}. Available periods: {', '.join(HISTORICAL_REFERENCES.keys())}"
        }
    
    # Return references based on perspective
    if perspective == "round_earth":
        return {
            "status": "success",
            "reference": HISTORICAL_REFERENCES[normalized_period]["round_earth"],
            "period": normalized_period
        }
    elif perspective == "flat_earth":
        return {
            "status": "success",
            "reference": HISTORICAL_REFERENCES[normalized_period]["flat_earth"],
            "period": normalized_period
        }
    else:
        # Return both perspectives
        return {
            "status": "success",
            "reference": {
                "round_earth": HISTORICAL_REFERENCES[normalized_period]["round_earth"],
                "flat_earth": HISTORICAL_REFERENCES[normalized_period]["flat_earth"]
            },
            "period": normalized_period
        }

def image_reference(category: str, tool_context: ToolContext = None) -> dict:
    """
    Retrieves image references related to Earth shape theories.
    
    Args:
        category (str): Image category (e.g., "space_photos", "horizon_test")
        tool_context (ToolContext, optional): The tool context for state management
    
    Returns:
        dict: Image reference data
    """
    perspective = ""
    # Track usage in state if context is provided
    if tool_context:
        categories_queried = tool_context.state.get("categories_queried", [])
        if category not in categories_queried:
            categories_queried.append(category)
            tool_context.state["categories_queried"] = categories_queried
        
        # Try to infer perspective from agent name in tool_context
        # In the ADK, we can infer this from the calling agent via tool_context
        if tool_context:
            calling_agent = getattr(tool_context, 'calling_agent', None) or ''
            if "flat" in calling_agent.lower():
                perspective = "flat_earth"
            elif "round" in calling_agent.lower():
                perspective = "round_earth"
    
    # Normalize category input
    normalized_category = category.lower().strip()
    
    # Check if category exists
    if normalized_category not in IMAGE_REFERENCES:
        return {
            "status": "error",
            "error_message": f"No image references found for category: {category}. Available categories: {', '.join(IMAGE_REFERENCES.keys())}"
        }
    
    # Return image references based on perspective
    if perspective == "round_earth":
        return {
            "status": "success",
            "image": IMAGE_REFERENCES[normalized_category]["round_earth"],
            "category": normalized_category
        }
    elif perspective == "flat_earth":
        return {
            "status": "success",
            "image": IMAGE_REFERENCES[normalized_category]["flat_earth"],
            "category": normalized_category
        }
    else:
        # Return both perspectives
        return {
            "status": "success",
            "image": {
                "round_earth": IMAGE_REFERENCES[normalized_category]["round_earth"],
                "flat_earth": IMAGE_REFERENCES[normalized_category]["flat_earth"]
            },
            "category": normalized_category
        }
