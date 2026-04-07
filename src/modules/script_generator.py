"""
Script generation module for podcast creation.
Handles initial script generation and length correction using Gemini API.
"""

from google import genai
from google.genai import types
import re
import time

from ..config import HOST_A_PERSONA, HOST_B_PERSONA, CUSTOM_INSTRUCTIONS, DEFAULT_TARGET_WORD_LENGTH, THINKING_BUDGET, THINKING_BUDGET_SCRIPGEN, SCRIPT_LENGTH_TOLERANCE, DEFAULT_MODEL
from ..models.schemas import GeneratedScript, ScriptOutput, TokenUsage
from ..utils.logging_utils import LoggingManager


class ScriptGenerator:
    """Handles podcast script generation from PDF content."""
    
    def __init__(self, client, logger: LoggingManager, prompt_logger=None):
        self.client = client
        self.logger = logger
        self.prompt_logger = prompt_logger
    
    def validate_host_alternation(self, script: str) -> tuple[bool, list[str]]:
        """
        Validate that Host A and Host B properly alternate in the script.
        
        Args:
            script: The script text to validate
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        if '\\n' in script and script.count('\\n') > script.count('\n'):
            script = script.replace('\\n', '\n')
        lines = script.strip().split('\n')
        host_lines = []
        violations = []
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line and (line.startswith('Host A:') or line.startswith('Host B:')):
                host_lines.append((i, line[:7]))  # Store line number and "Host A:" or "Host B:"
        
        if len(host_lines) < 2:
            return True, []  # Can't have alternation issues with less than 2 host lines
        
        prev_host = None
        for line_num, current_host in host_lines:
            if prev_host == current_host:
                violations.append(f"Line {line_num}: Consecutive {current_host} lines detected")
            prev_host = current_host
        
        return len(violations) == 0, violations
    
    def _count_words(self, text: str) -> int:
        """Count actual spoken words, excluding SSML tags and markup."""
        lines = text.strip().split('\n')
        content_lines = []
        
        for line in lines:
            if line.strip().startswith(('Host A:', 'Host B:')):
                content = line.split(':', 1)[1].strip()
                content_lines.append(content)
            else:
                content_lines.append(line.strip())
        
        full_content = ' '.join(content_lines)
        
        cleaned_content = re.sub(r'<[^>]*>', '', full_content)
        
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()
        
        words = re.findall(r"\b[\w\u3130-\u318F\uAC00-\uD7AF]+(?:'[\w\u3130-\u318F\uAC00-\uD7AF]+)?\b", cleaned_content)
        return len(words)
    
    def _build_script_generation_prompt(
        self, 
        target_word_length: int = DEFAULT_TARGET_WORD_LENGTH,
        additional_instructions: str = None,
        host_a_persona: str = None,
        host_b_persona: str = None,
        custom_instructions: str = None
    ) -> str:
        """Build the prompt for initial script generation."""
        
        actual_host_a = host_a_persona if host_a_persona else HOST_A_PERSONA
        actual_host_b = host_b_persona if host_b_persona else HOST_B_PERSONA  
        actual_custom_instructions = custom_instructions if custom_instructions else CUSTOM_INSTRUCTIONS
        
        if custom_instructions:
            print(f"🔍 Using custom instructions: '{actual_custom_instructions[:50]}...'")

        static_base = f"""You are a scriptwriter for a popular radiology podcast called "SummaryCast." Your task is to generate a podcast script based on a provided source text (e.g., a summary of a research paper or a technical article). The script must meticulously replicate the specific conversational style, structure, and host personas of the "SummaryCast" podcast.

The podcast has two AI hosts, Host A (female) and Host B (male). Their dynamic is crucial. It is not an interview; it is a collaborative unpacking of a complex topic for the listener.

CUSTOM INSTRUCTIONS are given by the client, and while you should respect them, all other rules and guidelines come first. 

--- CONVERSATIONAL STYLE GUIDELINES ---
Conversational Fillers: Use "uh," "um," "you know," "like," "hmm", and "well" naturally to avoid sounding overly scripted.
The Rule of Two: As a strict rule, no host should speak for more than two or three consecutive sentences without an interjection, question, or reaction from the other host. The dialogue must be a volley, not a series of speeches.
Break Down the Monologue: Find the key pieces of information in the source text (e.g., the problem, the method, the result) and break them down into a rapid back-and-forth.
Example of what to AVOID (Heavy and Slow):
    Host B: They tested it thoroughly across multiple data sets from their own institution, UAB, and an external one, UCSF, to ensure generalizability. They used four main performance metrics: first was matching findings to the correct impression in the top five results, second was matching reports of the same type using a MAP score, third was retrieving reports based on free text queries, and fourth was testing if the model could help an LLM make better diagnoses in a RAG setup.
Example of what to CREATE (Dynamic and Engaging):
    Host B: They were pretty thorough. They tested it across multiple data sets, um, both internal from their place, UAB, and external from UCSF...
    Host A: Good to test generalizability.
    Host B: Right. And they used four main ways to measure performance.
    Host A: Okay. What were they?
    Host B: First, matching findings to the correct impression. Could it find the right conclusion in the top five results...
...and so on. The information is the same, but the delivery is broken up into a dynamic conversation.

--- SCRIPT STRUCTURE GUIDE ---
Write a two-host podcast script about an academic paper (original research or review). Begin with a creative welcome that clearly opens the show at the very start. From there, let the conversation flow naturally without naming any segments: you might open with a striking observation or question, briefly situate the topic in real life, contrast with common practice, surface what the paper contributes, describe how it was investigated (design, data, comparisons), bring out the most important evidence (including memorable numbers), interpret what it means, acknowledge caveats, relate to prior work, note ethical or equity considerations, and highlight practical takeaways or future directions—only those that fit the specific paper. Use plain language for jargon when needed, weave figures into dialogue instead of listing them, and keep momentum with occasional micro-recaps only if helpful. End with a concise closing thought that distills the takeaways and a clear sign-off.

--- DO NOTS ---
IMPORTANT!! **The hosts must NOT impersonate humans. The hosts must NOT appear to be talking about a human experience from a first-person view.**
Do NOT invite listeners to "contact us" or subscribe.
Do NOT give the hosts names.

--- HOST CONFIGURATION ---
**Host Personas:**
- Host A: {actual_host_a}
- Host B: {actual_host_b}

--- CUSTOM INSTRUCTIONS ---
{actual_custom_instructions}"""

        variable_suffix = f"""

--- TARGET SPECIFICATIONS ---
**Target Word Length:** {target_word_length} words
Do your best to be as close to this target as possible."""

        if additional_instructions:
            variable_suffix += f"""

--- REVISION GUIDANCE ---
{additional_instructions}"""
        
        full_prompt = static_base + variable_suffix

        return full_prompt

    def _build_length_correction_prompt(
        self, 
        script: str, 
        current_word_count: int, 
        target_word_length: int,
        feedback: str,
        host_a_persona: str = None,
        host_b_persona: str = None,
        custom_instructions: str = None
    ) -> str:
        """Build prompt for script length correction with static elements first."""
        
        actual_host_a = host_a_persona if host_a_persona else HOST_A_PERSONA
        actual_host_b = host_b_persona if host_b_persona else HOST_B_PERSONA
        actual_custom_instructions = custom_instructions if custom_instructions else CUSTOM_INSTRUCTIONS
        
        static_base = f"""Revise the script to meet the target word count, based on the provided feedback, while preserving the core content and conversational style.

--- HOST CONFIGURATION ---
**Host Personas:**
- Host A: {actual_host_a}
- Host B: {actual_host_b}

--- CUSTOM INSTRUCTIONS ---
{actual_custom_instructions}"""
        
        variable_suffix = f"""

--- CURRENT SCRIPT ---
{script}

--- LENGTH ANALYSIS ---
**Current Word Count:** {current_word_count}
**Target Word Count:** {target_word_length}

--- REVISION FEEDBACK ---
{feedback}"""
        
        return static_base + variable_suffix
    
    def generate_initial_script(
        self, 
        cache_name: str, 
        target_word_length: int = DEFAULT_TARGET_WORD_LENGTH
    ) -> ScriptOutput:
        """
        Generate initial podcast script from cached PDF content.
        
        Args:
            cache_name: Name of the cached PDF content
            target_word_length: Target word count for the script
            
        Returns:
            ScriptOutput with generated script and metadata
        """
        start_time = time.time()
        prompt = self._build_script_generation_prompt(target_word_length)
        
        if cache_name.startswith("files/"):
            file_obj = self.client.files.get(name=cache_name)
            response = self.client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=[file_obj, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeneratedScript,
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET_SCRIPGEN)
                )
            )
        else:
            response = self.client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeneratedScript,
                    cached_content=cache_name,
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET_SCRIPGEN)
                )
            )
        
        if self.prompt_logger:
            self.prompt_logger.log_interaction(
                stage="script_generation",
                prompt=prompt,
                response=response,
                metadata={"target_word_length": target_word_length}
            )
        
        script_text = response.parsed.script_text
        word_count = self._count_words(script_text)
        
        execution_time = time.time() - start_time
        
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            cached_tokens = getattr(usage, 'cached_content_token_count', None)
            if cached_tokens is None:
                cached_tokens = 0
            token_usage = TokenUsage(
                prompt_tokens=usage.prompt_token_count,
                cached_tokens=cached_tokens,
                candidate_tokens=usage.candidates_token_count,
                total_tokens=usage.total_token_count
            )
            self.logger.log_stage_metrics("script_generation", token_usage, DEFAULT_MODEL, execution_time=execution_time)
        
        estimated_duration = word_count / 150.0
        
        return ScriptOutput(
            script_text=script_text,
            word_count=word_count,
            estimated_duration_minutes=estimated_duration
        )
    
    def correct_script_length(
        self, 
        script: str, 
        target_word_length: int,
        cache_name: str
    ) -> ScriptOutput:
        """
        Correct script length if it's outside the target range.
        
        Args:
            script: Current script text
            target_word_length: Target word count
            cache_name: Cache name for PDF content access
            
        Returns:
            ScriptOutput with length-corrected script
        """
        current_word_count = self._count_words(script)
        target_min = int(target_word_length * (1 - SCRIPT_LENGTH_TOLERANCE))
        target_max = int(target_word_length * (1 + SCRIPT_LENGTH_TOLERANCE))
        
        if target_min <= current_word_count <= target_max:
            estimated_duration = current_word_count / 150.0
            return ScriptOutput(
                script_text=script,
                word_count=current_word_count,
                estimated_duration_minutes=estimated_duration
            )
        
        if current_word_count < target_min:
            deficit = target_word_length - current_word_count
            feedback = f"Script is too short by {deficit} words. Add more detailed explanations, examples, or expand on key concepts while maintaining conversational flow."
        else:
            excess = current_word_count - target_word_length
            feedback = f"Script is too long by {excess} words. Condense content, remove redundant explanations, or streamline dialogue while preserving key information."
        
        start_time = time.time()
        prompt = self._build_length_correction_prompt(
            script, current_word_count, target_word_length, feedback
        )
        
        if cache_name.startswith("files/"):
            file_obj = self.client.files.get(name=cache_name)
            response = self.client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=[file_obj, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeneratedScript,
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
                )
            )
        else:
            response = self.client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeneratedScript,
                    cached_content=cache_name,
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
                )
            )
        
        if self.prompt_logger:
            self.prompt_logger.log_interaction(
                stage="script_length_correction",
                prompt=prompt,
                response=response,
                metadata={
                    "current_word_count": current_word_count,
                    "target_word_length": target_word_length,
                    "feedback": feedback
                }
            )
        
        corrected_script = response.parsed.script_text
        corrected_word_count = self._count_words(corrected_script)
        
        execution_time = time.time() - start_time
        
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            cached_tokens = getattr(usage, 'cached_content_token_count', None)
            if cached_tokens is None:
                cached_tokens = 0
            token_usage = TokenUsage(
                prompt_tokens=usage.prompt_token_count,
                cached_tokens=cached_tokens,
                candidate_tokens=usage.candidates_token_count,
                total_tokens=usage.total_token_count
            )
            self.logger.log_stage_metrics("script_length_correction", token_usage, DEFAULT_MODEL, execution_time=execution_time)
        
        estimated_duration = corrected_word_count / 150.0
        
        return ScriptOutput(
            script_text=corrected_script,
            word_count=corrected_word_count,
            estimated_duration_minutes=estimated_duration
        )
    
    def generate_script_with_length_correction(
        self,
        cache_name: str,
        target_word_length: int = DEFAULT_TARGET_WORD_LENGTH,
        max_iterations: int = 3,
        additional_instructions: str = None,
        host_a_persona: str = None,
        host_b_persona: str = None,
        custom_instructions: str = None
    ) -> ScriptOutput:
        """
        Generate script with chat-style length correction using Pattern A.
        Starts in chat from the first generation to avoid double-generation.
        """
        print(f"  → Generating script with chat-based length correction (target: {target_word_length} words)...")
        start_time = time.time()
        
        chat = self._create_chat_session(cache_name)
        
        generation_prompt = self._build_script_generation_prompt(
            target_word_length, additional_instructions, host_a_persona, host_b_persona, custom_instructions
        )
        if cache_name.startswith("files/"):
            file_obj = self.client.files.get(name=cache_name)
            response = chat.send_message([file_obj, generation_prompt])
        else:
            response = chat.send_message(generation_prompt)
        
        if self.prompt_logger:
            self.prompt_logger.log_interaction(
                stage="script_generation",
                prompt=generation_prompt,
                response=response,
                metadata={"target_word_length": target_word_length}
            )
        
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            cached_tokens = getattr(usage, 'cached_content_token_count', None) or 0
            token_usage = TokenUsage(
                prompt_tokens=usage.prompt_token_count,
                cached_tokens=cached_tokens,
                candidate_tokens=usage.candidates_token_count,
                total_tokens=usage.total_token_count
            )
            execution_time = time.time() - start_time
            self.logger.log_stage_metrics("script_generation", token_usage, DEFAULT_MODEL, execution_time=execution_time)
        
        current_script_text = response.parsed.script_text
        word_count = self._count_words(current_script_text)
        print(f"  → Initial script generated: {word_count} words")
        
        is_valid, violations = self.validate_host_alternation(current_script_text)
        if not is_valid:
            print(f"  ⚠️ Host alternation violations detected in initial script:")
            for violation in violations:
                print(f"    - {violation}")
            print("  → Regenerating script due to host alternation violations...")
            
            alternation_feedback = (
                "The script has host alternation issues. Host A and Host B must strictly alternate - "
                "no consecutive lines from the same host. Please regenerate the script ensuring "
                "proper Host A/Host B alternation throughout."
            )
            response = chat.send_message(alternation_feedback)
            current_script_text = response.parsed.script_text
            word_count = self._count_words(current_script_text)
            print(f"  → Regenerated script: {word_count} words")
        
        target_min = int(target_word_length * (1 - SCRIPT_LENGTH_TOLERANCE))
        target_max = int(target_word_length * (1 + SCRIPT_LENGTH_TOLERANCE))
        print(f"  → Target range: {target_min}-{target_max} words")
        
        attempts = [(current_script_text, word_count, 0)]  # (script, word_count, iteration)
        
        iteration = 0
        while iteration < max_iterations:
            is_length_valid = target_min <= word_count <= target_max
            is_alternation_valid, violations = self.validate_host_alternation(current_script_text)
            
            if is_length_valid and is_alternation_valid:
                break  # Both conditions satisfied
            
            print(f"  → Iteration {iteration + 1}: Current word count: {word_count}")
            
            feedback_parts = []
            
            if not is_length_valid:
                feedback_parts.append(f"Current word count: {word_count}. Target range: {target_min}-{target_max} words.")
            
            if not is_alternation_valid:
                feedback_parts.append("Host alternation issues detected: Host A and Host B must strictly alternate - no consecutive lines from the same host.")
                print(f"  ⚠️ Host alternation violations:")
                for violation in violations:
                    print(f"    - {violation}")
            
            feedback = " ".join(feedback_parts) + " Please revise the previous script accordingly."
            print(f"  → Sending revision request...")
            
            response = chat.send_message(feedback)
            current_script_text = response.parsed.script_text
            word_count = self._count_words(current_script_text)
            
            attempts.append((current_script_text, word_count, iteration + 1))
            
            if self.prompt_logger:
                self.prompt_logger.log_interaction(
                    stage=f"script_length_revision_iter_{iteration + 1}",
                    prompt=feedback,
                    response=response,
                    metadata={
                        "previous_word_count": word_count,
                        "new_word_count": self._count_words(current_script_text),
                        "target_range": f"{target_min}-{target_max}",
                        "iteration": iteration + 1
                    }
                )
            
            iteration += 1
        
        final_length_valid = target_min <= word_count <= target_max
        final_alternation_valid, final_violations = self.validate_host_alternation(current_script_text)
        
        if final_length_valid and final_alternation_valid:
            print(f"  ✓ Script meets all requirements after {iteration} iterations")
        elif iteration >= max_iterations:
            print(f"  → Max iterations reached. Selecting best attempt...")
            
            def calculate_distance(script_text, word_count):
                """Calculate distance from target length, prioritizing alternation validity"""
                _, violations = self.validate_host_alternation(script_text)
                alternation_penalty = len(violations) * 1000  # Heavy penalty for alternation issues
                length_distance = abs(word_count - target_word_length)
                return alternation_penalty + length_distance
            
            best_attempt = min(attempts, key=lambda x: calculate_distance(x[0], x[1]))
            best_script, best_word_count, best_iteration = best_attempt
            
            current_script_text = best_script
            word_count = best_word_count
            
            print(f"  → Selected attempt from iteration {best_iteration}: {best_word_count} words")
            
            final_length_valid = target_min <= word_count <= target_max
            final_alternation_valid, final_violations = self.validate_host_alternation(current_script_text)
            
            status_parts = []
            if not final_length_valid:
                status_parts.append(f"word count: {word_count}")
            if not final_alternation_valid:
                status_parts.append("host alternation issues")
            
            if status_parts:
                print(f"  → Best attempt still has: {', '.join(status_parts)}")
            else:
                print(f"  ✓ Best attempt meets all requirements")
        else:
            status_parts = []
            if not final_length_valid:
                status_parts.append(f"word count: {word_count}")
            if not final_alternation_valid:
                status_parts.append("host alternation issues")
            print(f"  → Script has remaining issues: {', '.join(status_parts)}")
        
        if self.prompt_logger:
            conversation_history = []
            for message in chat.get_history():
                if hasattr(message.parts[0], 'text'):
                    content = message.parts[0].text
                elif hasattr(message.parts[0], 'inline_data'):
                    content = "[File attachment]"
                else:
                    content = str(message.parts[0])
                
                conversation_history.append({
                    "role": message.role,
                    "content": content
                })
            
            self.prompt_logger.log_interaction(
                stage="script_generation_chat_full",
                prompt="Full conversation for script generation with length revision",
                response=None,
                metadata={
                    "conversation_history": conversation_history,
                    "final_word_count": word_count,
                    "target_range": f"{target_min}-{target_max}",
                    "iterations_used": iteration
                }
            )
        
        estimated_duration = word_count / 150.0
        
        return ScriptOutput(
            script_text=current_script_text,
            word_count=word_count,
            estimated_duration_minutes=estimated_duration
        )
    
    def _create_chat_session(self, cache_name: str):
        """Create a chat session with appropriate configuration."""
        if cache_name.startswith("files/"):
            return self.client.chats.create(
                model=DEFAULT_MODEL,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeneratedScript,
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET_SCRIPGEN)
                )
            )
        else:
            return self.client.chats.create(
                model=DEFAULT_MODEL,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeneratedScript,
                    cached_content=cache_name,
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET_SCRIPGEN)
                )
            )