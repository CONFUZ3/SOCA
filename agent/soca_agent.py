"""
SOCAAgent — drop-in replacement for ConversationManager using Google ADK.

Public interface is identical to ConversationManager so app.py changes are
minimal:
  - chat(user_message, conversation_history, problem_state, uploaded_data_summary)
  - notify_data_uploaded(conversation_history, problem_state, uploaded_data_summary)

Internal flow:
  1. Before each Runner.run_async() call, set up the thread-local state bridge so
     that tool functions can read/write GeoDataFrames in Streamlit session state.
  2. Sync problem_state primitives into the ADK session state.
  3. Call Runner.run_async() (async generator) via asyncio.run() and collect events.
  4. Extract the final text response.
  5. Sync updated session state back into problem_state.
  6. Return {response, actions, updated_state, tool_calls} matching the old interface.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from config.settings import settings
from agent.adk_prompts import build_adk_instruction
from agent.tools.fetch_tools import fetch_city_data
from agent.tools.optimize_tools import stage_optimization, confirm_optimization
from agent.tools.status_tools import get_data_status
from agent.tools.state_bridge import set_current_context
from agent.prompts import build_data_summary_text

logger = logging.getLogger(__name__)

_APP_NAME = "soca"
_USER_ID = "soca_user"


class SOCAAgent:
    """ADK-based conversational agent for spatial optimization.

    Replaces ConversationManager.  Tools handle all data fetching and
    optimization dispatching — no regex parsing of LLM output.
    """

    def __init__(self, api_key: str, problem_registry) -> None:
        self.api_key = api_key
        self.problem_registry = problem_registry

        self._session_service = InMemorySessionService()
        self._runner: Optional[Runner] = None

    # ------------------------------------------------------------------
    # Public interface (matches ConversationManager)
    # ------------------------------------------------------------------

    def chat(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        problem_state: Dict[str, Any],
        uploaded_data_summary: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Send a user message and return the agent response.

        Returns:
            {
                "response": str,        # agent's text reply
                "actions": [],          # always empty — tools handle side-effects
                "updated_state": Dict,  # updated problem_state
                "tool_calls": list[str] # names of ADK tools that were called
            }
        """
        try:
            return self._run(
                user_message=user_message,
                conversation_history=conversation_history,
                problem_state=problem_state,
                uploaded_data_summary=uploaded_data_summary,
            )
        except Exception as exc:
            logger.error("SOCAAgent.chat error: %s", exc, exc_info=True)
            err_msg = self._friendly_error(exc)
            return {
                "response": err_msg,
                "actions": [],
                "updated_state": problem_state,
                "tool_calls": [],
            }

    async def chat_stream(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        problem_state: Dict[str, Any],
        uploaded_data_summary: Optional[Dict] = None,
    ):
        """Async generator yielding structured events as the turn progresses.

        Event shapes (dict):
          {"type": "tool_call_start",  "name": str, "args": dict}
          {"type": "tool_call_result", "name": str, "summary": Any}
          {"type": "token",            "text": str}
          {"type": "final",            "text": str, "tool_calls": list[str]}
          {"type": "error",            "message": str}

        Designed for FastAPI SSE: the handler translates each dict into an
        SSE frame. On exception the generator emits an ``error`` event then
        returns — it never raises to the caller.
        """
        try:
            runner = self._get_runner()
            session = await self._get_or_create_session_async(problem_state)
            self._sync_state_to_session(session, problem_state)

            set_current_context(
                data=problem_state.get("data", {}),
                problem_state=problem_state,
                problem_registry=self.problem_registry,
                generated_sites_count=problem_state.get("_generated_sites_count", 100),
                generated_sites_seed=problem_state.get("_generated_sites_seed"),
                network_manager=problem_state.get("_network_manager"),
            )

            message = self._build_message(
                user_message,
                conversation_history,
                problem_state,
                uploaded_data_summary,
            )

            tool_calls: list[str] = []
            text_parts: list[str] = []

            async for event in runner.run_async(
                user_id=_USER_ID,
                session_id=session.id,
                new_message=message,
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        fc = getattr(part, "function_call", None)
                        if fc is not None:
                            name = getattr(fc, "name", None)
                            args = getattr(fc, "args", None) or {}
                            try:
                                args = dict(args)
                            except Exception:
                                args = {}
                            if name:
                                tool_calls.append(name)
                                yield {
                                    "type": "tool_call_start",
                                    "name": name,
                                    "args": args,
                                }

                        fr = getattr(part, "function_response", None)
                        if fr is not None:
                            name = getattr(fr, "name", None)
                            response = getattr(fr, "response", None)
                            summary: Any
                            try:
                                summary = dict(response) if response else {}
                            except Exception:
                                summary = {"raw": str(response)}
                            yield {
                                "type": "tool_call_result",
                                "name": name,
                                "summary": summary,
                            }

                        text = getattr(part, "text", None)
                        if text:
                            is_final = True
                            if hasattr(event, "is_final_response"):
                                try:
                                    is_final = bool(event.is_final_response())
                                except Exception:
                                    is_final = True
                            if is_final:
                                text_parts.append(text)
                                yield {"type": "token", "text": text}

            refreshed = await self._session_service.get_session(
                app_name=_APP_NAME,
                user_id=_USER_ID,
                session_id=session.id,
            )
            if refreshed:
                self._sync_state_from_session(refreshed, problem_state)

            final_text = "\n".join(text_parts) if text_parts else (
                "I processed your request. Check the map and data panel for updates."
            )
            yield {
                "type": "final",
                "text": final_text,
                "tool_calls": tool_calls,
            }
        except Exception as exc:
            logger.error("SOCAAgent.chat_stream error: %s", exc, exc_info=True)
            yield {"type": "error", "message": self._friendly_error(exc)}

    def notify_data_uploaded(
        self,
        conversation_history: List[Dict[str, str]],
        problem_state: Dict[str, Any],
        uploaded_data_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Notify the agent that new data has been uploaded."""
        has_demand = False
        has_candidates = False
        for name, info in uploaded_data_summary.items():
            if (
                info.get("demand_columns")
                or "demand" in name.lower()
                or any(w in name.lower() for w in ["population", "people", "residents"])
            ):
                has_demand = True
            elif (
                any(w in name.lower() for w in ["candidate", "site", "facility", "location"])
                or info.get("capacity_columns")
                or info.get("cost_columns")
            ):
                has_candidates = True

        notice = (
            "System notice: New data has been uploaded or fetched. "
            "Please infer dataset roles (demand vs candidate sites vs boundary) "
            "from filenames, geometry types, and column schemas. "
            "Do NOT ask for confirmation of role inferences. "
            "Do NOT infer or set the optimization problem type; wait for the user. "
        )
        if has_demand and not has_candidates:
            notice += (
                "No candidate sites detected. The system will auto-generate "
                "100 random candidate sites within the demand extent when "
                "optimization runs. Mention this to the user. "
            )
        notice += (
            "Summarise your role inferences and ask what the user would like to do next."
        )

        return self.chat(
            user_message=notice,
            conversation_history=conversation_history,
            problem_state=problem_state,
            uploaded_data_summary=uploaded_data_summary,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_runner(self) -> Runner:
        if self._runner is None:
            self._runner = self._build_runner()
        return self._runner

    def _build_runner(self) -> Runner:
        problems_metadata = self.problem_registry.list_problems()
        for meta in problems_metadata:
            sn = meta.get("short_name")
            prob = self.problem_registry.get_problem(sn)
            if prob:
                meta["conversation_prompts"] = prob.get_conversation_prompts()

        instruction = build_adk_instruction(problems_metadata)

        agent = LlmAgent(
            name="soca_agent",
            model=settings.GEMINI_MODEL,
            instruction=instruction,
            tools=[fetch_city_data, stage_optimization, confirm_optimization, get_data_status],
            generate_content_config=genai_types.GenerateContentConfig(
                temperature=0.7,
            ),
        )

        return Runner(
            agent=agent,
            session_service=self._session_service,
            app_name=_APP_NAME,
        )

    async def _get_or_create_session_async(self, problem_state: Dict[str, Any]):
        """Async version — for use inside a running event loop (FastAPI path)."""
        session_id = problem_state.get("_adk_session_id")

        if session_id:
            session = await self._session_service.get_session(
                app_name=_APP_NAME,
                user_id=_USER_ID,
                session_id=session_id,
            )
            if session:
                return session

        session = await self._session_service.create_session(
            app_name=_APP_NAME,
            user_id=_USER_ID,
        )
        problem_state["_adk_session_id"] = session.id
        logger.debug("SOCAAgent: created new ADK session %s", session.id)
        return session

    def _get_or_create_session(self, problem_state: Dict[str, Any]):
        """Sync wrapper — for use outside an event loop (Streamlit path)."""
        return asyncio.run(self._get_or_create_session_async(problem_state))

    def _sync_state_to_session(
        self, session, problem_state: Dict[str, Any]
    ) -> None:
        """Push JSON-serialisable fields from problem_state into ADK session."""
        state = session.state
        state["problem_type"] = problem_state.get("problem_type")
        state["parameters"] = problem_state.get("parameters", {})
        state["constraints"] = problem_state.get("constraints", {})

    def _sync_state_from_session(
        self, session, problem_state: Dict[str, Any]
    ) -> None:
        """Pull updated fields from ADK session back into problem_state."""
        state = session.state
        if state.get("problem_type"):
            problem_state["problem_type"] = state["problem_type"]
        if state.get("parameters"):
            problem_state["parameters"] = dict(state["parameters"])
        if state.get("constraints"):
            problem_state["constraints"] = dict(state["constraints"])

    def _build_message(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        problem_state: Dict[str, Any],
        uploaded_data_summary: Optional[Dict],
    ) -> genai_types.Content:
        """Build the Content object passed to Runner.run_async() as new_message."""
        context_parts = [f"User message: {user_message}"]

        aoi = problem_state.get("aoi")
        if aoi:
            context_parts.append(
                f"\nUser AOI (already defined, reuse as boundary): name='{aoi.get('name')}', "
                f"area={aoi.get('area_km2', 0):.1f} km², source={aoi.get('source')}."
            )

        if problem_state.get("problem_type"):
            context_parts.append(
                f"\nCurrent problem type: {problem_state['problem_type']}"
            )
        if problem_state.get("parameters"):
            import json
            context_parts.append(
                f"Current parameters: {json.dumps(problem_state['parameters'])}"
            )
        if problem_state.get("solution"):
            sol = problem_state["solution"]
            context_parts.append(
                f"Solution available: status={sol.get('status')}, "
                f"objective={sol.get('objective_value', 'N/A')}"
            )

        if uploaded_data_summary:
            context_parts.append(
                "\nAvailable data:\n" + build_data_summary_text(uploaded_data_summary)
            )
        elif problem_state.get("data"):
            context_parts.append(
                f"\nDatasets loaded: {list(problem_state['data'].keys())}"
            )

        full_text = "\n".join(context_parts)

        return genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=full_text)],
        )

    async def _collect_events(
        self, runner: Runner, session_id: str, message: genai_types.Content
    ):
        """Async generator consumer — collect all events and track tool calls."""
        events = []
        tool_calls = []
        async for event in runner.run_async(
            user_id=_USER_ID,
            session_id=session_id,
            new_message=message,
        ):
            events.append(event)
            # Extract tool call names for UX feedback
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        name = getattr(part.function_call, "name", None)
                        if name:
                            tool_calls.append(name)
        return events, tool_calls

    def _extract_text_from_events(self, events: list) -> str:
        """Return the final text response from the event stream."""
        text_parts = []
        for event in events:
            # Skip non-final events when the method is present
            if hasattr(event, "is_final_response") and not event.is_final_response():
                continue
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)
        return "\n".join(text_parts) if text_parts else ""

    def _run(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        problem_state: Dict[str, Any],
        uploaded_data_summary: Optional[Dict],
    ) -> Dict[str, Any]:
        runner = self._get_runner()
        session = self._get_or_create_session(problem_state)

        self._sync_state_to_session(session, problem_state)

        set_current_context(
            data=problem_state.get("data", {}),
            problem_state=problem_state,
            problem_registry=self.problem_registry,
            generated_sites_count=problem_state.get("_generated_sites_count", 100),
            generated_sites_seed=problem_state.get("_generated_sites_seed"),
            network_manager=problem_state.get("_network_manager"),
        )

        message = self._build_message(
            user_message, conversation_history, problem_state, uploaded_data_summary
        )

        logger.info("SOCAAgent: running turn for session %s", session.id)
        try:
            events, tool_calls = asyncio.run(
                self._collect_events(runner, session.id, message)
            )
        except Exception as exc:
            logger.error("SOCAAgent: runner failed: %s", exc, exc_info=True)
            raise

        response_text = self._extract_text_from_events(events)
        if not response_text:
            response_text = "I processed your request. Please check the map and data panel for updates."

        refreshed_session = asyncio.run(self._session_service.get_session(
            app_name=_APP_NAME,
            user_id=_USER_ID,
            session_id=session.id,
        ))
        if refreshed_session:
            self._sync_state_from_session(refreshed_session, problem_state)

        logger.info(
            "SOCAAgent: turn complete, response length=%d, tool_calls=%s, "
            "solution=%s, data_keys=%s",
            len(response_text),
            tool_calls,
            problem_state.get("solution") is not None,
            list(problem_state.get("data", {}).keys()),
        )

        return {
            "response": response_text,
            "actions": [],
            "updated_state": problem_state,
            "tool_calls": tool_calls,
        }

    @staticmethod
    def _friendly_error(exc: Exception) -> str:
        msg = str(exc)
        if "api key" in msg.lower():
            return "API key issue detected. Please check your Gemini API key configuration."
        if "quota" in msg.lower() or "limit" in msg.lower():
            return "API quota exceeded. Please try again later."
        if "network" in msg.lower() or "connection" in msg.lower():
            return "Network connection issue. Please check your internet connection and try again."
        return f"I encountered an error: {msg}. Please try again."
