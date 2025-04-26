"""
Flask API Integration Demo

This script shows how to integrate the Science Information Retrieval Agent
with a simple Flask API for web access.
"""

import os
from datetime import date
from flask import Flask, request, jsonify
from pydantic import ValidationError

from science_agent import ScienceAgentConfig, get_science_info_sync

app = Flask(__name__)

# Check for API keys during startup
required_env_vars = {
    "OPENAI_API_KEY": "OpenAI API key is required to run this application."
}

for var, message in required_env_vars.items():
    if not os.environ.get(var):
        print(f"WARNING: {message}")


@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy", "version": "1.0.0"})


@app.route("/science/query", methods=["POST"])
def science_query():
    """
    Process a scientific query and return structured information.
    
    Expected JSON payload:
    {
        "query": "Scientific query text",
        "detail_level": 5,  # Optional, default is 5
        "max_references": 3,  # Optional, default is 3
        "include_limitations": true  # Optional, default is true
    }
    """
    try:
        # Get and validate request data
        data = request.json
        if not data or "query" not in data:
            return jsonify({"error": "Missing 'query' field in request"}), 400
        
        # Extract query and optional parameters
        query = data["query"]
        detail_level = int(data.get("detail_level", 5))
        max_references = int(data.get("max_references", 3))
        include_limitations = bool(data.get("include_limitations", True))
        
        # Validate input ranges
        if not (1 <= detail_level <= 10):
            return jsonify({"error": "detail_level must be between 1 and 10"}), 400
        if not (0 <= max_references <= 10):
            return jsonify({"error": "max_references must be between 0 and 10"}), 400
            
        # Create agent configuration
        config = ScienceAgentConfig(
            detail_level=detail_level,
            max_references=max_references,
            include_limitations=include_limitations,
            current_date=date.today()
        )
        
        # Process the query
        result = get_science_info_sync(query, config)
        
        # Convert Pydantic model to dictionary for JSON response
        response = result.dict()
        return jsonify(response)
        
    except ValidationError as e:
        # Handle Pydantic validation errors
        return jsonify({"error": "Validation error", "details": str(e)}), 400
    except Exception as e:
        # Handle other exceptions
        return jsonify({"error": str(e)}), 500


@app.route("/science/models", methods=["GET"])
def available_models():
    """Return a list of available LLM models for the agent"""
    # This is a simplified example - in a real app, this would be dynamic
    models = [
        {
            "id": "openai:gpt-4o",
            "name": "GPT-4o",
            "provider": "OpenAI",
            "capabilities": ["Structured output", "Tool usage"]
        },
        {
            "id": "anthropic:claude-3-5-sonnet-latest",
            "name": "Claude 3.5 Sonnet",
            "provider": "Anthropic",
            "capabilities": ["Structured output", "Tool usage"]
        },
        {
            "id": "google-gla:gemini-1.5-pro",
            "name": "Gemini 1.5 Pro",
            "provider": "Google",
            "capabilities": ["Structured output", "Tool usage"]
        }
    ]
    return jsonify(models)


@app.route("/", methods=["GET"])
def index():
    """Simple HTML documentation for the API"""
    docs = """
    <html>
    <head>
        <title>Science Information API</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #333; }
            h2 { color: #0066cc; margin-top: 30px; }
            pre { background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }
            .endpoint { margin-bottom: 30px; }
        </style>
    </head>
    <body>
        <h1>Science Information API</h1>
        <p>This API provides structured scientific information using LLMs.</p>
        
        <div class="endpoint">
            <h2>POST /science/query</h2>
            <p>Get structured information about a scientific topic.</p>
            <h3>Request:</h3>
            <pre>
{
  "query": "Explain quantum computing",
  "detail_level": 5,
  "max_references": 3,
  "include_limitations": true
}
            </pre>
            <h3>Response:</h3>
            <p>Returns a structured JSON response with scientific information.</p>
        </div>
        
        <div class="endpoint">
            <h2>GET /science/models</h2>
            <p>List available LLM models for the agent.</p>
        </div>
        
        <div class="endpoint">
            <h2>GET /health</h2>
            <p>Health check endpoint.</p>
        </div>
    </body>
    </html>
    """
    return docs


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    
    # In production, you would use a proper WSGI server
    # For demo purposes, we use the built-in development server
    app.run(host="0.0.0.0", port=port, debug=True)
