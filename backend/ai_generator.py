import json
import re
from openai import OpenAI
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with DeepSeek API for generating responses"""

    MAX_TOOL_ROUNDS = 2  # Maximum sequential tool-call rounds per query

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **Up to two sequential searches per query** — use a second search only when the first result is insufficient, or the question spans two distinct topics/courses
- Do NOT search twice for the same query string — vary parameters meaningfully
- Synthesize ALL search results into a single coherent response
- If search yields no results, state this clearly without offering alternatives
- IMPORTANT: Always use the provided function calling interface. Never output function calls as raw text or XML markup.

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]

        # Prepare API call parameters
        api_params = {
            **self.base_params,
            "messages": messages,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"

        # Get response from DeepSeek
        response = self.client.chat.completions.create(**api_params)

        # Handle tool execution if needed
        if response.choices[0].finish_reason == "tool_calls" and tool_manager:
            return self._handle_tool_execution_loop(response, messages, tools, tool_manager)

        # Fallback: detect DeepSeek native DSML function call format in text content
        content = response.choices[0].message.content
        if tool_manager and content and '<｜DSML｜function_calls>' in content:
            dsml_call = self._parse_dsml_tool_call(content)
            if dsml_call:
                return self._handle_dsml_tool_execution(dsml_call, messages, tool_manager)

        return content or ""

    def _execute_tool_calls(self, assistant_message, messages: List, tool_manager) -> bool:
        """
        Execute all tool calls in an assistant message, appending results to messages.

        Returns:
            True if at least one tool call succeeded, False if all failed.
        """
        had_any_success = False
        for tool_call in assistant_message.tool_calls:
            try:
                tool_result = tool_manager.execute_tool(
                    tool_call.function.name,
                    **json.loads(tool_call.function.arguments)
                )
                had_any_success = True
            except Exception as e:
                tool_result = f"Tool execution failed: {str(e)}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })
        return had_any_success

    def _handle_tool_execution_loop(
        self, initial_response, messages: List, tools: Optional[List], tool_manager
    ) -> str:
        """
        Execute up to MAX_TOOL_ROUNDS sequential tool-call rounds.

        Each round: execute tools → call API with tools → check if another round needed.
        After the loop, one final API call without tools synthesizes the answer.
        """
        current_response = initial_response
        round_num = 0

        while round_num < self.MAX_TOOL_ROUNDS:
            assistant_message = current_response.choices[0].message
            messages.append(assistant_message)

            had_any_success = self._execute_tool_calls(assistant_message, messages, tool_manager)

            # On total failure, skip remaining rounds and go straight to synthesis
            if not had_any_success:
                break

            # Call API with tools so the model can decide whether to search again
            next_params = {
                **self.base_params,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
            }
            next_response = self.client.chat.completions.create(**next_params)

            if next_response.choices[0].finish_reason != "tool_calls":
                # Model chose not to call another tool; append its message and stop looping
                messages.append(next_response.choices[0].message)
                break

            current_response = next_response
            round_num += 1

        # Final synthesis call without tools
        final_params = {**self.base_params, "messages": messages}
        final_response = self.client.chat.completions.create(**final_params)
        final_content = final_response.choices[0].message.content

        if tool_manager and final_content and '<｜DSML｜function_calls>' in final_content:
            dsml_call = self._parse_dsml_tool_call(final_content)
            if dsml_call:
                return self._handle_dsml_tool_execution(dsml_call, messages, tool_manager)

        return final_content or ""

    def _parse_dsml_tool_call(self, content: str) -> Optional[Dict]:
        """Parse DeepSeek native DSML function call format embedded in text content"""
        invoke_match = re.search(
            r'<｜DSML｜invoke name="([^"]+)">(.*?)</｜DSML｜invoke>',
            content, re.DOTALL
        )
        if not invoke_match:
            return None

        tool_name = invoke_match.group(1)
        params_block = invoke_match.group(2)

        params = {}
        for param_name, param_value in re.findall(
            r'<｜DSML｜parameter name="([^"]+)"[^>]*>(.*?)</｜DSML｜parameter>',
            params_block, re.DOTALL
        ):
            value = param_value.strip()
            # Convert numeric strings to int for parameters like lesson_number
            if value.lstrip('-').isdigit():
                value = int(value)
            params[param_name] = value

        return {"tool_name": tool_name, "params": params}

    def _handle_dsml_tool_execution(self, dsml_call: Dict, messages: List, tool_manager) -> str:
        """Execute a tool call parsed from DSML format and get a follow-up response"""
        tool_result = tool_manager.execute_tool(dsml_call["tool_name"], **dsml_call["params"])

        messages.append({
            "role": "user",
            "content": f"Search results:\n{tool_result}\n\nPlease answer the original question using these results."
        })

        final_params = {**self.base_params, "messages": messages}
        final_response = self.client.chat.completions.create(**final_params)
        return final_response.choices[0].message.content
