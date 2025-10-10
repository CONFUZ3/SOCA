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
        self.model = genai.GenerativeModel("gemini-flash-lite-latest")
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
        Send message to Gemini with full context.
        
        Returns:
        {
            "response": str,  # Gemini's text response
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
                "gemini-flash-lite-latest",
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
            return self._parse_response(response, problem_state, last_user_message=user_message)
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}", exc_info=True)
            
            # Provide more specific error messages
            if "API key" in str(e).lower():
                error_msg = "API key issue detected. Please check your Gemini API key configuration."
            elif "quota" in str(e).lower() or "limit" in str(e).lower():
                error_msg = "API quota exceeded. Please try again later or check your API limits."
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                error_msg = "Network connection issue. Please check your internet connection and try again."
            else:
                error_msg = f"I encountered an error: {str(e)}. Please try again."
            
            return {
                "response": error_msg,
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
        Gemini has no memory between requests.
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
        problem_state: Dict[str, Any],
        last_user_message: Optional[str] = None
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

                # Normalize problem type and parameters for robustness
                action_data = self._normalize_action(action_data)
                
                # Merge any parameters from problem state that might be missing from action
                if action_data.get("action") == "optimize":
                    current_params = action_data.get('parameters', {})
                    state_params = problem_state.get('parameters', {})
                    # Merge state parameters into action parameters (action takes precedence)
                    merged_params = {**state_params, **current_params}
                    action_data['parameters'] = merged_params
                
                # Validate action
                if action_data.get("action") == "optimize":
                    # Validate required fields
                    if "problem_type" in action_data and "parameters" in action_data:
                        # Validate variant-specific parameters
                        validation_error = self._validate_variant_parameters(action_data)
                        if validation_error:
                            logger.warning(f"Variant parameter validation failed: {validation_error}")
                            # Try to add default parameters for missing variant requirements
                            action_data = self._add_default_variant_parameters(action_data)
                        # Gate on explicit parameter confirmation from the user
                        is_confirmed = bool(action_data.get('confirm')) or bool(problem_state.get('parameters_confirmed'))
                        if not is_confirmed and self._is_affirmative(last_user_message or ""):
                            is_confirmed = True

                        # Always update problem state with proposed values for visibility
                        problem_state['problem_type'] = action_data['problem_type']
                        problem_state['parameters'] = action_data.get('parameters', {})
                        problem_state['constraints'] = action_data.get('constraints', {})
                        logger.info(f"Conversation Manager: Action parameters: {action_data.get('parameters', {})}")

                        if is_confirmed:
                            actions.append(action_data)
                            text_response = "I'm ready to solve your problem. Let me run the optimization..."
                        else:
                            # Ask for confirmation of parameters without asking about data roles
                            summary = json.dumps({
                                "problem_type": problem_state.get('problem_type'),
                                "parameters": problem_state.get('parameters', {}),
                                "constraints": problem_state.get('constraints', {})
                            }, indent=2)
                            text_response = (
                                "Please confirm these parameters before I optimize (data roles have already been inferred):\n\n"
                                f"```json\n{summary}\n```\n\nReply with 'yes' to proceed or update any values."
                            )
                            problem_state['pending_action'] = action_data
                            problem_state['parameters_confirmed'] = False
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

        # If the user has just confirmed and a pending action exists, pass it through
        if updated_state.get('parameters_confirmed') and updated_state.get('pending_action'):
            actions.append(updated_state['pending_action'])
            # Clear pending action after promoting it
            updated_state.pop('pending_action', None)
        
        return {
            "response": text_response,
            "actions": actions,
            "updated_state": updated_state
        }

    def notify_data_uploaded(
        self,
        conversation_history: List[Dict[str, str]],
        problem_state: Dict[str, Any],
        uploaded_data_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Notify the model that new data has been uploaded and ask it to
        infer dataset roles without requesting user confirmation.
        """
        notice = (
            "System notice: New data uploaded. Please infer dataset roles "
            "(e.g., demand vs. candidate sites) based on filenames, geometry types, "
            "and schemas. Proceed without asking the user to confirm these inferences. "
            "Do NOT infer or set the optimization problem type; wait for the user to specify it. "
            "Summarize your data-role inferences and suggest what parameters the user might provide next."
        )
        return self.chat(
            user_message=notice,
            conversation_history=conversation_history,
            problem_state=problem_state,
            uploaded_data_summary=uploaded_data_summary
        )
    
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
        
        # Do NOT infer problem type from assistant responses. Only infer from explicit user input
        # The caller (_parse_response) handles confirmation gating and passes through actions.
        
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

        # Detect MCLP variant mentions
        variants = [
            ("budget", ["budget"]),
            ("capacitated", ["capacitated", "capacity", "capacities"]),
            ("probabilistic", ["probabilistic", "reliability", "failure"]),
            ("multi_coverage", ["multi-coverage", "multi coverage", "k-coverage", "k coverage"]),
            ("backup", ["backup"]) ,
            ("classical", ["classical"]) 
        ]
        for variant_key, keywords in variants:
            if any(kw in response_lower for kw in keywords):
                if 'parameters' not in updated_state:
                    updated_state['parameters'] = {}
                updated_state['parameters']['variant'] = variant_key
                break

        # Extract budget if mentioned
        budget_match = re.search(r'budget\s*(of|=)?\s*(\d+\.?\d*)', response_lower)
        if budget_match:
            budget_val = float(budget_match.group(2))
            if 'parameters' not in updated_state:
                updated_state['parameters'] = {}
            updated_state['parameters']['budget'] = budget_val

        # Extract k-coverage like "k=2" or "at least 2 facilities"
        k_match = re.search(r'k\s*=?\s*(\d+)', response_lower)
        if k_match:
            k_val = int(k_match.group(1))
            if 'parameters' not in updated_state:
                updated_state['parameters'] = {}
            updated_state['parameters']['k_coverage'] = k_val
        else:
            atleast_match = re.search(r'at\s+least\s+(\d+)\s+(facilities|sites|centers?)', response_lower)
            if atleast_match:
                k_val = int(atleast_match.group(1))
                if 'parameters' not in updated_state:
                    updated_state['parameters'] = {}
                updated_state['parameters']['k_coverage'] = k_val

        # Extract capacity information
        capacity_match = re.search(r'capacity\s*(of|=)?\s*(\d+\.?\d*)', response_lower)
        if capacity_match:
            capacity_val = float(capacity_match.group(2))
            if 'parameters' not in updated_state:
                updated_state['parameters'] = {}
            # If capacities not already set, create a list with this value
            if 'capacities' not in updated_state['parameters']:
                updated_state['parameters']['capacities'] = [capacity_val]
            else:
                updated_state['parameters']['capacities'].append(capacity_val)

        # Extract budget information
        budget_match = re.search(r'budget\s*(of|=)?\s*(\d+\.?\d*)', response_lower)
        if budget_match:
            budget_val = float(budget_match.group(2))
            if 'parameters' not in updated_state:
                updated_state['parameters'] = {}
            updated_state['parameters']['budget'] = budget_val
        
        # If model text indicates confirmation (rare), set flag
        if self._is_affirmative(response_text):
            updated_state['parameters_confirmed'] = True
        return updated_state

    def _is_affirmative(self, text: str) -> bool:
        """Heuristic check for user/model confirmation."""
        t = text.strip().lower()
        if not t:
            return False
        affirmatives = ["yes", "y", "confirm", "proceed", "go ahead", "ok", "okay", "do it"]
        return any(a == t or t.startswith(a) for a in affirmatives)

    def _validate_variant_parameters(self, action: Dict[str, Any]) -> Optional[str]:
        """Validate that variant-specific parameters are present."""
        problem_type = action.get('problem_type', '').lower()
        parameters = action.get('parameters', {})
        
        if problem_type == 'mclp':
            variant = parameters.get('variant', 'classical')
            
            if variant == 'capacitated':
                if 'capacities' not in parameters:
                    return "Capacitated MCLP requires 'capacities' parameter"
            elif variant == 'budget':
                if 'budget' not in parameters:
                    return "Budget MCLP requires 'budget' parameter"
            elif variant in ['multi_coverage', 'backup']:
                if 'k_coverage' not in parameters:
                    return f"{variant.title()} MCLP requires 'k_coverage' parameter"
        
        return None  # No validation errors
    
    def _add_default_variant_parameters(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Add default parameters for missing variant requirements."""
        problem_type = action.get('problem_type', '').lower()
        parameters = action.get('parameters', {})
        
        if problem_type == 'mclp':
            variant = parameters.get('variant', 'classical')
            
            if variant == 'capacitated' and 'capacities' not in parameters:
                # Add default capacities - this will be handled by the solver
                logger.info("Adding default capacities for capacitated MCLP variant")
                # The solver will handle default capacity calculation
                
            elif variant == 'budget' and 'budget' not in parameters:
                # Add default budget based on number of facilities
                n_facilities = parameters.get('n_facilities', 5)
                default_budget = n_facilities * 1000  # Default cost per facility
                parameters['budget'] = default_budget
                logger.info(f"Adding default budget of {default_budget} for budget MCLP variant")
                
            elif variant in ['multi_coverage', 'backup'] and 'k_coverage' not in parameters:
                # Add default k_coverage
                default_k = 2  # Both multi_coverage and backup require at least 2 facilities
                parameters['k_coverage'] = default_k
                logger.info(f"Adding default k_coverage of {default_k} for {variant} MCLP variant")
        
        action['parameters'] = parameters
        return action
    
    def _normalize_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize problem_type and parameter names/values for model robustness."""
        normalized = dict(action)
        # Normalize problem type to registered short names when possible
        pt = (normalized.get('problem_type') or '').lower()
        if pt and pt not in self.problem_registry._problems:
            # Try to infer using registry if not an exact short name
            inferred = self.problem_registry.infer_problem_type(pt)
            if inferred:
                normalized['problem_type'] = inferred
        # Parameter normalization
        params = dict(normalized.get('parameters') or {})
        # Map common synonyms
        if 'service_distance' in params and 'service_radius' not in params:
            params['service_radius'] = params.pop('service_distance')
        if 'distance_threshold' in params and 'service_radius' not in params:
            params['service_radius'] = params.pop('distance_threshold')
        if 'k' in params and 'k_coverage' not in params:
            params['k_coverage'] = params.pop('k')
        if 'reliability' in params and 'facility_reliability' not in params:
            params['facility_reliability'] = params.pop('reliability')
        # Normalize variant
        if 'variant' in params and isinstance(params['variant'], str):
            params['variant'] = params['variant'].lower().replace('-', '_').strip()
        normalized['parameters'] = params
        return normalized

