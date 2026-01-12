import google.generativeai as genai
from typing import List, Dict, Any, Optional
import json
import logging
import numpy as np
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
                        # Always require explicit confirmation before optimization
                        is_confirmed = bool(action_data.get('confirm')) or bool(problem_state.get('parameters_confirmed'))
                        if not is_confirmed and self._is_affirmative(last_user_message or ""):
                            is_confirmed = True

                        # Always update problem state with proposed values for visibility
                        problem_state['problem_type'] = action_data['problem_type']
                        problem_state['parameters'] = action_data.get('parameters', {})
                        problem_state['constraints'] = action_data.get('constraints', {})
                        logger.info(f"Conversation Manager: Action parameters: {action_data.get('parameters', {})}")

                        if is_confirmed:
                            # Only add action if we don't already have one
                            if len(actions) == 0:
                                actions.append(action_data)
                                text_response = "I'm ready to solve your problem. Let me run the optimization..."
                            else:
                                logger.warning("Skipping duplicate action - action already exists in actions list")
                                text_response = "I'm ready to solve your problem. Let me run the optimization..."
                        else:
                            # Ask for confirmation of parameters - always require explicit confirmation
                            summary = json.dumps({
                                "problem_type": problem_state.get('problem_type'),
                                "parameters": problem_state.get('parameters', {}),
                                "constraints": problem_state.get('constraints', {})
                            }, indent=2)
                            text_response = (
                                "Please confirm these parameters before I optimize:\n\n"
                                f"```json\n{summary}\n```\n\n"
                                "**Important:** I will only run the optimization after you explicitly confirm with 'yes' or 'proceed'. "
                                "You can modify any parameters by simply stating the changes (e.g., 'change n_facilities to 5' or 'set budget to 1000'). "
                                "Please review the parameters carefully and reply with 'yes' to proceed or specify any changes."
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
        # Process both assistant response and user message for parameter updates
        updated_state = self._extract_state_updates(text_response, problem_state)
        if last_user_message:
            user_updates = self._extract_state_updates(last_user_message, updated_state)
            updated_state.update(user_updates)

        # If the user has just confirmed and a pending action exists, pass it through
        # BUT only if we haven't already added an action from the JSON parsing above
        if updated_state.get('parameters_confirmed') and updated_state.get('pending_action') and len(actions) == 0:
            actions.append(updated_state['pending_action'])
            # Clear pending action after promoting it
            updated_state.pop('pending_action', None)
        
        # Log final actions for debugging
        if actions:
            logger.info(f"Conversation Manager: Returning {len(actions)} action(s): {[a.get('action', 'unknown') for a in actions]}")
        
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
        # Check if we have demand data but no candidate sites
        has_demand = False
        has_candidates = False
        
        for name, data_info in uploaded_data_summary.items():
            # Check if this looks like demand data
            if (data_info.get('demand_columns') or 
                'demand' in name.lower() or 
                any(word in name.lower() for word in ['population', 'people', 'residents'])):
                has_demand = True
            # Check if this looks like candidate data
            elif (any(word in name.lower() for word in ['candidate', 'site', 'facility', 'location']) or
                  data_info.get('capacity_columns') or
                  data_info.get('cost_columns')):
                has_candidates = True
        
        notice = (
            "System notice: New data uploaded. Please infer dataset roles "
            "(e.g., demand vs. candidate sites) based on filenames, geometry types, "
            "and schemas. Proceed without asking the user to confirm these inferences. "
            "Do NOT infer or set the optimization problem type; wait for the user to specify it. "
        )
        
        # Add candidate generation notice if needed
        if has_demand and not has_candidates:
            notice += (
                "IMPORTANT: No candidate sites detected. The system will automatically "
                "generate 100 random candidate sites within the demand extent when optimization runs. "
                "Users can adjust the count and set a random seed in the sidebar. "
                "Inform the user about this automatic generation feature. "
            )
        
        notice += (
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
        facility_patterns = [
            r'(\d+)\s+facilities?',
            r'n_facilities[:\s=]+(\d+)',
            r'number\s+of\s+facilities[:\s=]+(\d+)',
            r'locate\s+(\d+)\s+facilities?',
            r'place\s+(\d+)\s+facilities?',
            r'(\d+)\s+sites?',
            r'(\d+)\s+stations?',
            r'(\d+)\s+centers?',
            r'change\s+n_facilities\s+to\s+(\d+)',
            r'set\s+n_facilities\s+to\s+(\d+)',
            r'n_facilities\s+to\s+(\d+)'
        ]
        for pattern in facility_patterns:
            facility_match = re.search(pattern, response_lower)
            if facility_match:
                n_facilities = int(facility_match.group(1))
                if 'parameters' not in updated_state:
                    updated_state['parameters'] = {}
                updated_state['parameters']['n_facilities'] = n_facilities
                break
        
        # Check for service radius mentions - capture both value and unit
        # Patterns with explicit units (capture unit in group 2)
        radius_patterns_with_unit = [
            r'within\s+(\d+\.?\d*)\s*(km|kilometers?|mi|miles?|m|meters?)',
            r'(\d+\.?\d*)\s*(km|kilometers?|mi|miles?|m|meters?)\s+radius',
            r'radius\s+of\s+(\d+\.?\d*)\s*(km|kilometers?|mi|miles?|m|meters?)',
            r'service_radius[:\s=]+(\d+\.?\d*)\s*(km|kilometers?|mi|miles?|m|meters?)',
            r'service\s+radius[:\s=]+(\d+\.?\d*)\s*(km|kilometers?|mi|miles?|m|meters?)',
            r'distance\s+threshold[:\s=]+(\d+\.?\d*)\s*(km|kilometers?|mi|miles?|m|meters?)',
            r'(\d+\.?\d*)\s*(km|kilometers?|mi|miles?|m|meters?)\s+service',
        ]
        # Patterns without explicit units (value only)
        radius_patterns_no_unit = [
            r'radius\s+of\s+(\d+\.?\d*)',
            r'service_radius[:\s=]+(\d+\.?\d*)',
            r'service\s+radius[:\s=]+(\d+\.?\d*)',
            r'distance\s+threshold[:\s=]+(\d+\.?\d*)',
            r'maximum\s+distance[:\s=]+(\d+\.?\d*)',
            r'max\s+distance[:\s=]+(\d+\.?\d*)',
        ]
        
        radius_found = False
        # First try patterns with units
        for pattern in radius_patterns_with_unit:
            radius_match = re.search(pattern, response_lower)
            if radius_match:
                value = float(radius_match.group(1))
                unit = radius_match.group(2).lower()
                
                # Normalize unit and convert to meters
                service_radius_meters, normalized_unit = self._convert_radius_to_meters(value, unit)
                
                if 'parameters' not in updated_state:
                    updated_state['parameters'] = {}
                updated_state['parameters']['service_radius'] = service_radius_meters
                updated_state['parameters']['service_radius_original'] = value
                updated_state['parameters']['service_radius_unit'] = 'm'  # Value is already in meters
                radius_found = True
                logger.info(f"Parsed service radius: {value} {normalized_unit} = {service_radius_meters} meters")
                break
        
        # If no unit pattern matched, try patterns without units
        if not radius_found:
            for pattern in radius_patterns_no_unit:
                radius_match = re.search(pattern, response_lower)
                if radius_match:
                    service_radius = float(radius_match.group(1))
                    if 'parameters' not in updated_state:
                        updated_state['parameters'] = {}
                    updated_state['parameters']['service_radius'] = service_radius
                    # No unit info - will be determined later by distance calculator
                    logger.info(f"Parsed service radius: {service_radius} (no unit specified)")
                    break

        # Detect variant mentions - only from explicit user requests, not data descriptions
        # Only set variant if user explicitly mentions wanting that variant
        variants = [
            ("budget", ["budget variant", "budget constraint", "budget problem"]),
            ("capacitated", ["capacitated variant", "capacity constraint", "capacity problem", "facility capacity"]),
            ("probabilistic", ["probabilistic variant", "reliability problem"]),
            ("multi_coverage", ["multi-coverage variant", "k-coverage variant"]),
            ("backup", ["backup variant"]) ,
            ("classical", ["classical variant", "standard variant"]) 
        ]
        for variant_key, keywords in variants:
            if any(kw in response_lower for kw in keywords):
                if 'parameters' not in updated_state:
                    updated_state['parameters'] = {}
                updated_state['parameters']['variant'] = variant_key
                break

        # Extract budget if mentioned - but only if user is explicitly requesting budget variant
        # Check if user is talking about budget variant first
        budget_variant_mentioned = any(phrase in response_lower for phrase in [
            'budget variant', 'budget constraint', 'budget problem', 'budget optimization',
            'with budget', 'budget limit', 'cost constraint', 'budget constraint'
        ])
        
        if budget_variant_mentioned:
            budget_patterns = [
                r'budget\s*(of|=|:)?\s*(\d+\.?\d*)',
                r'budget[:\s=]+(\d+\.?\d*)',
                r'total\s+budget[:\s=]+(\d+\.?\d*)',
                r'cost\s+limit[:\s=]+(\d+\.?\d*)',
                r'maximum\s+cost[:\s=]+(\d+\.?\d*)',
                r'change\s+budget\s+to\s+(\d+\.?\d*)',
                r'set\s+budget\s+to\s+(\d+\.?\d*)',
                r'budget\s+to\s+(\d+\.?\d*)'
            ]
            for pattern in budget_patterns:
                budget_match = re.search(pattern, response_lower)
                if budget_match:
                    # Handle different group patterns
                    if len(budget_match.groups()) >= 2 and budget_match.group(2):
                        budget_val = float(budget_match.group(2))
                    else:
                        budget_val = float(budget_match.group(1))
                    if 'parameters' not in updated_state:
                        updated_state['parameters'] = {}
                    updated_state['parameters']['budget'] = budget_val
                    # DO NOT automatically set variant - only set if explicitly requested by user
                    # The variant should only be set when the user explicitly mentions "budget variant" or similar
                    break

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

        # Extract max assignment distance for P-Median variants
        max_dist_patterns = [
            r'max_assignment_distance[:\s=]+(\d+\.?\d*)',
            r'max\s+assignment\s+distance[:\s=]+(\d+\.?\d*)',
            r'maximum\s+assignment\s+distance[:\s=]+(\d+\.?\d*)',
            r'assignment\s+distance\s+limit[:\s=]+(\d+\.?\d*)',
            r'max\s+travel\s+distance[:\s=]+(\d+\.?\d*)',
            r'maximum\s+travel\s+distance[:\s=]+(\d+\.?\d*)'
        ]
        for pattern in max_dist_patterns:
            max_dist_match = re.search(pattern, response_lower)
            if max_dist_match:
                max_dist_val = float(max_dist_match.group(1))
                if 'parameters' not in updated_state:
                    updated_state['parameters'] = {}
                updated_state['parameters']['max_assignment_distance'] = max_dist_val
                break
        
        # Handle parameter clearing/reset requests
        if any(word in response_lower for word in ['clear', 'reset', 'remove', 'delete']):
            if 'parameters' in response_lower:
                # Clear all parameters
                updated_state['parameters'] = {}
                logger.info("User requested to clear all parameters")
            elif 'variant' in response_lower:
                # Clear variant
                if 'parameters' in updated_state:
                    updated_state['parameters'].pop('variant', None)
                logger.info("User requested to clear variant")
        
        # Handle specific parameter removal
        if 'remove' in response_lower or 'delete' in response_lower:
            if 'n_facilities' in response_lower and 'parameters' in updated_state:
                updated_state['parameters'].pop('n_facilities', None)
            if 'budget' in response_lower and 'parameters' in updated_state:
                updated_state['parameters'].pop('budget', None)
            if 'service_radius' in response_lower and 'parameters' in updated_state:
                updated_state['parameters'].pop('service_radius', None)
        
        # If model text indicates confirmation (rare), set flag
        if self._is_affirmative(response_text):
            updated_state['parameters_confirmed'] = True
        
        # Final safeguard: Remove budget parameter if no budget variant is explicitly set
        if 'parameters' in updated_state and 'budget' in updated_state['parameters']:
            variant = updated_state['parameters'].get('variant', 'base')
            if variant not in ['budget']:
                logger.warning(f"Removing accidentally extracted budget parameter - variant is '{variant}', not 'budget'")
                updated_state['parameters'].pop('budget', None)
        
        return updated_state

    def _is_affirmative(self, text: str) -> bool:
        """Heuristic check for user/model confirmation."""
        t = text.strip().lower()
        if not t:
            return False
        affirmatives = ["yes", "y", "confirm", "proceed", "go ahead", "ok", "okay", "do it"]
        return any(a == t or t.startswith(a) for a in affirmatives)
    
    def _convert_radius_to_meters(self, value: float, unit: str) -> tuple:
        """
        Convert a radius value to meters based on the specified unit.
        
        Args:
            value: The numeric value
            unit: The unit string (e.g., 'km', 'kilometers', 'm', 'meters', 'mi', 'miles')
        
        Returns:
            Tuple of (value_in_meters, normalized_unit)
        """
        unit = unit.lower().strip()
        
        # Normalize unit variations
        if unit in ('km', 'kilometer', 'kilometers'):
            return value * 1000, 'km'
        elif unit in ('m', 'meter', 'meters'):
            return value, 'm'
        elif unit in ('mi', 'mile', 'miles'):
            return value * 1609.34, 'miles'
        else:
            # Unknown unit, assume meters
            logger.warning(f"Unknown unit '{unit}', assuming meters")
            return value, 'm'

    def _validate_variant_parameters(self, action: Dict[str, Any]) -> Optional[str]:
        """Validate that variant-specific parameters are present."""
        problem_type = action.get('problem_type', '').lower()
        parameters = action.get('parameters', {})
        
        if problem_type == 'p-median':
            variant = parameters.get('variant', 'base')
            
            if variant == 'capacitated':
                if 'capacities' not in parameters:
                    return "Capacitated P-Median requires 'capacities' parameter"
            elif variant == 'budget':
                if 'budget' not in parameters:
                    return "Budget P-Median requires 'budget' parameter"
                if 'facility_costs' not in parameters:
                    return "Budget P-Median requires 'facility_costs' parameter"
            elif variant == 'max_distance':
                if 'max_assignment_distance' not in parameters:
                    return "Max-distance P-Median requires 'max_assignment_distance' parameter"
        
        elif problem_type == 'mclp':
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
        
        if problem_type == 'p-median':
            variant = parameters.get('variant', 'base')
            
            if variant == 'capacitated' and 'capacities' not in parameters:
                # Add default capacities - this will be handled by the solver
                logger.info("Adding default capacities for capacitated P-Median variant")
                # The solver will handle default capacity calculation
                
            elif variant == 'budget' and 'budget' not in parameters:
                # Only add default budget if facility_costs are also available
                if 'facility_costs' in parameters and parameters['facility_costs']:
                    # Calculate a reasonable budget based on facility costs
                    facility_costs = parameters['facility_costs']
                    if isinstance(facility_costs, (list, np.ndarray)) and len(facility_costs) > 0:
                        # Set budget to allow selecting at least n_facilities
                        n_facilities = parameters.get('n_facilities', 5)
                        sorted_costs = sorted(facility_costs)
                        # Budget should be enough to select the cheapest n_facilities
                        default_budget = sum(sorted_costs[:n_facilities]) * 1.1  # 10% buffer
                        parameters['budget'] = default_budget
                        logger.info(f"Adding default budget of {default_budget} for budget P-Median variant based on facility costs")
                    else:
                        # No facility costs available, remove budget variant
                        logger.warning("Cannot add budget variant without facility_costs, reverting to base variant")
                        parameters['variant'] = 'base'
                else:
                    # No facility costs available, remove budget variant
                    logger.warning("Cannot add budget variant without facility_costs, reverting to base variant")
                    parameters['variant'] = 'base'
                
            elif variant == 'max_distance' and 'max_assignment_distance' not in parameters:
                # Add default max assignment distance
                default_distance = 10.0  # Default maximum assignment distance
                parameters['max_assignment_distance'] = default_distance
                logger.info(f"Adding default max_assignment_distance of {default_distance} for max-distance P-Median variant")
        
        elif problem_type == 'mclp':
            variant = parameters.get('variant', 'classical')
            
            if variant == 'capacitated' and 'capacities' not in parameters:
                # Add default capacities - this will be handled by the solver
                logger.info("Adding default capacities for capacitated MCLP variant")
                # The solver will handle default capacity calculation
                
            elif variant == 'budget' and 'budget' not in parameters:
                # Only add default budget if facility_costs are also available
                if 'facility_costs' in parameters and parameters['facility_costs']:
                    # Calculate a reasonable budget based on facility costs
                    facility_costs = parameters['facility_costs']
                    if isinstance(facility_costs, (list, np.ndarray)) and len(facility_costs) > 0:
                        # Set budget to allow selecting at least n_facilities
                        n_facilities = parameters.get('n_facilities', 5)
                        sorted_costs = sorted(facility_costs)
                        # Budget should be enough to select the cheapest n_facilities
                        default_budget = sum(sorted_costs[:n_facilities]) * 1.1  # 10% buffer
                        parameters['budget'] = default_budget
                        logger.info(f"Adding default budget of {default_budget} for budget MCLP variant based on facility costs")
                    else:
                        # No facility costs available, remove budget variant
                        logger.warning("Cannot add budget variant without facility_costs, reverting to classical variant")
                        parameters['variant'] = 'classical'
                else:
                    # No facility costs available, remove budget variant
                    logger.warning("Cannot add budget variant without facility_costs, reverting to classical variant")
                    parameters['variant'] = 'classical'
                
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
        
        # CRITICAL FIX: Prevent automatic variant inference
        # Only allow variants if explicitly requested by user or if variant-specific parameters are provided
        variant = params.get('variant')
        if variant and variant != 'base' and variant != 'classical':
            # Check if this variant was explicitly requested by looking for variant-specific parameters
            # If no variant-specific parameters are present, reset to base/classical
            if pt == 'p-median':
                if variant == 'budget' and 'budget' not in params and 'facility_costs' not in params:
                    logger.warning(f"Removing auto-inferred budget variant for P-Median - no budget parameters provided")
                    params['variant'] = 'base'
                elif variant == 'capacitated' and 'capacities' not in params:
                    logger.warning(f"Removing auto-inferred capacitated variant for P-Median - no capacity parameters provided")
                    params['variant'] = 'base'
                elif variant == 'max_distance' and 'max_assignment_distance' not in params:
                    logger.warning(f"Removing auto-inferred max_distance variant for P-Median - no max_assignment_distance parameter provided")
                    params['variant'] = 'base'
            elif pt == 'mclp':
                if variant == 'budget' and 'budget' not in params:
                    logger.warning(f"Removing auto-inferred budget variant for MCLP - no budget parameter provided")
                    params['variant'] = 'classical'
                elif variant == 'capacitated' and 'capacities' not in params:
                    logger.warning(f"Removing auto-inferred capacitated variant for MCLP - no capacity parameters provided")
                    params['variant'] = 'classical'
                elif variant in ['multi_coverage', 'backup'] and 'k_coverage' not in params:
                    logger.warning(f"Removing auto-inferred {variant} variant for MCLP - no k_coverage parameter provided")
                    params['variant'] = 'classical'
        
        normalized['parameters'] = params
        return normalized

