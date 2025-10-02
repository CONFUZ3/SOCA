import google.generativeai as genai
from typing import List, Dict, Any, Optional
import json
import logging
from .prompts import build_system_prompt, build_data_summary_text

logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Manages conversation with Gemini API following SDK best practices.
    
    Key Responsibilities:
    - Maintain complete conversation history (Gemini is stateless)
    - Include problem state in every request
    - Parse structured outputs (JSON when needed)
    - Error handling and retry logic
    """
    
    def __init__(self, api_key: str, problem_registry):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
        self.problem_registry = problem_registry
        self.max_tokens = 4096
    
    def chat(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        problem_state: Dict[str, Any],
        uploaded_data_summary: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Send message to Claude with full context.
        
        Returns:
        {
            "response": str,  # Claude's text response
            "actions": List[Dict],  # Any actions to take (e.g., trigger optimization)
            "updated_state": Dict  # Updated problem state
        }
        """
        
        # Build comprehensive system prompt
        problems_metadata = self.problem_registry.list_problems()
        
        # Enhance metadata with conversation prompts
        for prob_meta in problems_metadata:
            short_name = prob_meta['short_name']
            problem = self.problem_registry.get_problem(short_name)
            if problem:
                prob_meta['conversation_prompts'] = problem.get_conversation_prompts()
        
        system_prompt = build_system_prompt(problems_metadata)
        
        # Prepare messages with full history + current state
        messages = self._prepare_messages(
            user_message,
            conversation_history,
            problem_state,
            uploaded_data_summary
        )
        
        # Call Gemini API
        try:
            # Convert messages to Gemini format
            chat_history = []
            for msg in messages:
                if msg["role"] == "user":
                    chat_history.append({"role": "user", "parts": [msg["content"]]})
                elif msg["role"] == "assistant":
                    chat_history.append({"role": "model", "parts": [msg["content"]]})
            
            # Create a new model instance with system instruction
            model_with_system = genai.GenerativeModel(
                "gemini-2.0-flash-exp",
                system_instruction=system_prompt
            )
            
            # Start chat session
            chat = model_with_system.start_chat(history=chat_history)
            
            # Send the latest user message
            response = chat.send_message(
                user_message,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=0.7,
                )
            )
            
            # Parse response
            return self._parse_response(response, problem_state)
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {
                "response": f"I encountered an error connecting to the AI service: {str(e)}. Please try again.",
                "actions": [],
                "updated_state": problem_state
            }
    
    def _prepare_messages(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        problem_state: Dict[str, Any],
        uploaded_data_summary: Optional[Dict]
    ) -> List[Dict[str, str]]:
        """
        Prepares message array with full context.
        
        CRITICAL: Include COMPLETE conversation history and current state.
        Claude has no memory between requests.
        """
        
        messages = []
        
        # Add conversation history (excluding current message)
        for msg in conversation_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Build current message with embedded state
        current_content = f"User message: {user_message}\n\n"
        
        # Add problem state context
        current_content += "# Current Problem State\n\n"
        
        if problem_state.get('problem_type'):
            current_content += f"**Problem Type:** {problem_state['problem_type']}\n"
        else:
            current_content += "**Problem Type:** Not yet identified\n"
        
        if problem_state.get('parameters'):
            current_content += f"**Parameters Collected:** {json.dumps(problem_state['parameters'], indent=2)}\n"
        
        if problem_state.get('constraints'):
            current_content += f"**Constraints Defined:** {json.dumps(problem_state['constraints'], indent=2)}\n"
        
        # Add data summary
        if uploaded_data_summary:
            current_content += "\n# Uploaded Data Summary\n\n"
            current_content += build_data_summary_text(uploaded_data_summary)
        else:
            current_content += "\n**Note:** No data has been uploaded yet.\n"
        
        # Add previous solution context if exists
        if problem_state.get('solution'):
            solution = problem_state['solution']
            current_content += "\n# Current Solution\n\n"
            current_content += f"**Status:** {solution.get('status')}\n"
            if solution.get('objective_value') is not None:
                current_content += f"**Objective Value:** {solution.get('objective_value'):.2f}\n"
            if solution.get('metrics'):
                current_content += f"**Key Metrics:**\n"
                for key, value in list(solution['metrics'].items())[:5]:  # Show first 5 metrics
                    if isinstance(value, (int, float)):
                        current_content += f"  - {key}: {value:.2f}\n"
                    else:
                        current_content += f"  - {key}: {value}\n"
        
        messages.append({
            "role": "user",
            "content": current_content
        })
        
        return messages
    
    def _parse_response(
        self,
        response,
        problem_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parses Gemini's response and extracts actions.
        
        Detects:
        - JSON action requests (optimization trigger)
        - State updates (parameter collection)
        - Normal conversational responses
        """
        
        # Extract text from response
        text_response = response.text
        
        # Check for JSON action
        actions = []
        if "action" in text_response.lower() and "{" in text_response:
            try:
                # Extract JSON (may be wrapped in markdown code blocks)
                json_str = text_response
                
                # Handle markdown code blocks
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    # Try to find JSON in any code block
                    parts = json_str.split("```")
                    for part in parts:
                        if "{" in part and "}" in part:
                            json_str = part.strip()
                            break
                
                # Try to parse JSON
                action_data = json.loads(json_str)
                
                # Validate action
                if action_data.get("action") == "optimize":
                    # Validate required fields
                    if "problem_type" in action_data and "parameters" in action_data:
                        actions.append(action_data)
                        # Update problem state with confirmed values
                        problem_state['problem_type'] = action_data['problem_type']
                        problem_state['parameters'] = action_data.get('parameters', {})
                        problem_state['constraints'] = action_data.get('constraints', {})
                        
                        text_response = "I'm ready to solve your problem. Let me run the optimization..."
                    else:
                        logger.warning("Invalid optimize action: missing required fields")
                
            except json.JSONDecodeError as e:
                logger.debug(f"Could not parse JSON from response: {e}")
                # Not a valid action, treat as normal text
                pass
            except Exception as e:
                logger.error(f"Error parsing action: {e}")
                pass
        
        # Try to extract state updates from conversation (heuristic)
        updated_state = self._extract_state_updates(text_response, problem_state)
        
        return {
            "response": text_response,
            "actions": actions,
            "updated_state": updated_state
        }
    
    def _extract_state_updates(
        self,
        response_text: str,
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Attempts to extract parameter/constraint updates from conversation.
        
        This is a heuristic approach - looks for problem type mentions and
        parameter discussions to update state.
        """
        updated_state = current_state.copy()
        
        # Try to infer problem type if not set
        if not current_state.get('problem_type'):
            inferred_type = self.problem_registry.infer_problem_type(response_text)
            if inferred_type:
                updated_state['problem_type'] = inferred_type
                logger.info(f"Inferred problem type: {inferred_type}")
        
        # Look for parameter mentions (basic heuristic)
        # This could be enhanced with more sophisticated NLP
        response_lower = response_text.lower()
        
        # Check for facility count mentions
        import re
        facility_match = re.search(r'(\d+)\s+facilities?', response_lower)
        if facility_match:
            n_facilities = int(facility_match.group(1))
            if 'parameters' not in updated_state:
                updated_state['parameters'] = {}
            updated_state['parameters']['n_facilities'] = n_facilities
        
        # Check for service radius mentions
        radius_match = re.search(r'radius\s+of\s+(\d+\.?\d*)', response_lower)
        if radius_match:
            service_radius = float(radius_match.group(1))
            if 'parameters' not in updated_state:
                updated_state['parameters'] = {}
            updated_state['parameters']['service_radius'] = service_radius
        
        return updated_state

