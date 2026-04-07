"""
Quality assurance modules for podcast script evaluation and improvement.
Includes script judging, revision, fact-checking, and content appropriateness checks.
"""

from google import genai
from google.genai import types
from typing import List, Optional, Dict, Tuple
import re
import time

from ..config import HOST_A_PERSONA, HOST_B_PERSONA, CUSTOM_INSTRUCTIONS, THINKING_BUDGET, THINKING_BUDGET_EDITOR, DEFAULT_MODEL
from ..models.schemas import (
    JudgeOutput, ScriptLineEdits, ExtractedClaims, 
    VerifiedClaimsAssessment, InappropriatenessCheck, TokenUsage, GeneratedScript
)
from .script_generator import ScriptGenerator
from ..utils.logging_utils import LoggingManager


class QualityAssurance:
    """Handles all quality assurance operations for podcast scripts."""
    
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
    
    
    def _add_order_keys(self, script: str) -> Tuple[str, Dict[str, str]]:
        """Add simple numeric order keys to script lines."""
        lines = script.strip().split('\n')
        indexed_lines = []
        key_to_line = {}
        
        for i, line in enumerate(lines, 1):
            order_key = str(i)
            indexed_lines.append(f"[{order_key}] {line}")
            key_to_line[order_key] = line
        
        return '\n'.join(indexed_lines), key_to_line
    
    def _apply_line_edits(self, script: str, edits: ScriptLineEdits) -> tuple[str, list[str]]:
        """Apply line number based edits to a script with host validation.
        
        Returns:
            Tuple of (edited_script, validation_errors)
        """
        lines = script.strip().split('\n')
        validation_errors = []
        
        for edit in edits.edits:
            try:
                line_index = edit.line_number - 1
                if 0 <= line_index < len(lines):
                    original_line = lines[line_index]
                    order_key = None
                    if original_line.startswith('[') and '] ' in original_line:
                        key_end = original_line.find('] ')
                        if key_end > 0:
                            order_key = original_line[1:key_end]
                            original_content = original_line[key_end + 2:]
                        else:
                            original_content = original_line
                    else:
                        original_content = original_line
                    
                    original_content_stripped = original_content.strip()
                    edit_content_stripped = edit.content.strip()
                    
                    if original_content_stripped.startswith(('Host A:', 'Host B:')):
                        if not edit_content_stripped.startswith(('Host A:', 'Host B:')):
                            validation_errors.append(f"Line {edit.line_number}: Cannot remove host prefix from '{original_content_stripped[:20]}...'")
                            continue
                        
                        original_host = original_content_stripped[:7]  # "Host A:" or "Host B:"
                        new_host = edit_content_stripped[:7]
                        if original_host != new_host:
                            validation_errors.append(f"Line {edit.line_number}: Cannot change {original_host} to {new_host}")
                            continue
                    
                    if order_key:
                        lines[line_index] = f"[{order_key}] {edit.content}"
                    else:
                        lines[line_index] = edit.content
                else:
                    print(f"  → Warning: Line number {edit.line_number} is out of range")
            except Exception as e:
                print(f"  → Warning: Failed to apply edit on line {edit.line_number}: {e}")
                continue
        
        return '\n'.join(lines), validation_errors
    
    
    def _strip_order_keys(self, script: str) -> str:
        """Remove order keys from script to return clean output."""
        lines = script.strip().split('\n')
        clean_lines = []
        
        for line in lines:
            if line.strip() and line.startswith('[') and '] ' in line:
                key_end = line.find('] ')
                if key_end > 0:
                    clean_lines.append(line[key_end + 2:])
                else:
                    clean_lines.append(line)
            else:
                clean_lines.append(line)
        
        return '\n'.join(clean_lines)
    
    def rewrite_script_major(
        self, 
        script: str, 
        revision_feedback: str, 
        cache_name: str
    ) -> str:
        """
        Perform complete script rewrite for major revisions.
        
        Args:
            script: Current script text
            revision_feedback: Feedback for revision
            cache_name: Cache name for PDF access
            
        Returns:
            Completely rewritten script text
        """
        
        rewrite_prompt = f"""Generate a completely new podcast script based on the source material and the provided feedback about the current script.

--- FEEDBACK ON CURRENT SCRIPT ---
{revision_feedback}

--- CURRENT SCRIPT (for reference) ---
{script}

--- INSTRUCTIONS ---
Create a fresh script that addresses all the feedback above. Do not try to edit the current script - write a completely new version that incorporates the source material more effectively."""
        
        print(f"  → Performing major revision (complete rewrite)...")
        
        start_time = time.time()
        
        if cache_name.startswith("files/"):
            file_obj = self.client.files.get(name=cache_name)
            response = self.client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=[file_obj, rewrite_prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeneratedScript,
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
                )
            )
        else:
            response = self.client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=[rewrite_prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeneratedScript,
                    cached_content=cache_name,
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
                )
            )
        
        execution_time = time.time() - start_time
        rewritten_script = response.parsed.script_text
        
        is_valid, violations = self.validate_host_alternation(rewritten_script)
        if not is_valid:
            print(f"  ⚠️ Host alternation violations detected in rewritten script:")
            for violation in violations:
                print(f"    - {violation}")
            print("  → Regenerating script due to alternation violations...")
            
            enhanced_rewrite_prompt = rewrite_prompt + "\n\nCRITICAL: Ensure Host A and Host B strictly alternate throughout - no consecutive lines from the same host."
            
            if cache_name.startswith("files/"):
                file_obj = self.client.files.get(name=cache_name)
                response = self.client.models.generate_content(
                    model=DEFAULT_MODEL,
                    contents=[file_obj, enhanced_rewrite_prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GeneratedScript,
                        thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
                    )
                )
            else:
                response = self.client.models.generate_content(
                    model=DEFAULT_MODEL,
                    contents=[enhanced_rewrite_prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GeneratedScript,
                        cached_content=cache_name,
                        thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
                    )
                )
            
            rewritten_script = response.parsed.script_text
            print(f"  → Regenerated script with alternation fix")
        
        if self.prompt_logger:
            self.prompt_logger.log_interaction(
                stage="script_rewrite_major",
                prompt=rewrite_prompt,
                response=response,
                metadata={
                    "original_script_length": len(script),
                    "rewritten_script_length": len(rewritten_script),
                    "revision_feedback": revision_feedback
                }
            )
        
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            token_usage = TokenUsage(
                prompt_tokens=usage.prompt_token_count,
                cached_tokens=getattr(usage, 'cached_content_token_count', 0) or 0,
                candidate_tokens=usage.candidates_token_count,
                total_tokens=usage.total_token_count
            )
            self.logger.log_stage_metrics("script_rewrite_major", token_usage, DEFAULT_MODEL, execution_time=execution_time)
        
        return rewritten_script
    
    def judge_script(
        self, 
        script: str, 
        cache_name: str,
        previous_script: str = None,
        previous_feedback: str = None
    ) -> JudgeOutput:
        """
        Evaluate script quality against source material and criteria.
        
        Args:
            script: The script to evaluate
            cache_name: Cache name for PDF access
            previous_script: Previous version if this is a revision
            previous_feedback: Previous feedback if this is a revision
            
        Returns:
            JudgeOutput with evaluation results
        """
        static_base = f"""Critically evaluate the draft podcast script against the source article and quality criteria.
Identify any factual inaccuracies, omissions, issues with tone, style, educational value, or audience relevance.
**CAVEAT** Your OCR ability is not perfect. When the script and source disagree with a technical word, use COMMON SENSE to figure out what is correct. **IMPORTANT**
Provide specific feedback for improvement. YOU MUST BE EXTREMELY SPECIFIC WITH YOUR FEEDBACK. 
[IMPORTANT] Even if it looks like a minor issue, if you can't PINPOINT the changes SPECIFICALLY, it is to be Major revision!! Don't give minor revision with abstract suggestions.

--- EVALUATION GUIDELINES ---
- **Minor Revision**: Issues that can be fixed by changing words, phrases, or individual lines (maximum 5 lines to edit) without adding/removing content (most common). 
- **Major Revision**: Requires adding new dialogue, removing sections, restructuring conversation flow, or significant content changes
- **Accept**: Use sparingly for near-perfect scripts
- **Reject**: Use sparingly for fundamentally flawed scripts

If this is not the first evaluation (i.e., `previous_script` and `previous_feedback` are provided), assess if the revision adequately addressed the previous feedback.

--- CUSTOM INSTRUCTIONS ---
{CUSTOM_INSTRUCTIONS}"""
        
        variable_suffix = f"""

--- SCRIPT TO EVALUATE ---
{script}"""
        
        if previous_script and previous_feedback:
            variable_suffix += f"""

--- REVISION CONTEXT ---
**Previous Script:**
{previous_script}

**Previous Feedback:**
{previous_feedback}"""
        
        full_prompt = static_base + variable_suffix
        
        start_time = time.time()
        if cache_name.startswith("files/"):
            file_obj = self.client.files.get(name=cache_name)
            response = self.client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=[file_obj, full_prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=JudgeOutput,
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
                )
            )
        else:
            response = self.client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=[full_prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=JudgeOutput,
                    cached_content=cache_name,
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
                )
            )
        execution_time = time.time() - start_time
        
        if self.prompt_logger:
            self.prompt_logger.log_interaction(
                stage="qa_judge",
                prompt=full_prompt,
                response=response,
                metadata={"has_previous_feedback": previous_feedback is not None}
            )
        
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            token_usage = TokenUsage(
                prompt_tokens=usage.prompt_token_count,
                cached_tokens=getattr(usage, 'cached_content_token_count', 0) or 0,
                candidate_tokens=usage.candidates_token_count,
                total_tokens=usage.total_token_count
            )
            self.logger.log_stage_metrics("script_evaluation", token_usage, DEFAULT_MODEL, execution_time=execution_time)
        
        return response.parsed
    
    def revise_script_minor(
        self, 
        script: str, 
        revision_feedback: str, 
        cache_name: str
    ) -> str:
        """
        Perform minor script revision using line-based editing (REPLACE only).
        
        Args:
            script: Current script text
            revision_feedback: Feedback for revision
            cache_name: Cache name for PDF access
            
        Returns:
            Revised script text
        """
        indexed_script, self._key_mapping = self._add_order_keys(script)
        
        static_base = f"""Revise the podcast script by providing edits for specific lines that need word/phrase/line-level changes. Do not add or remove content - only modify existing lines.

Analyze the revision feedback and identify lines that need minor corrections. For each line that needs changes, provide an edit with the corrected content.
YOU MUST ONLY REVISE ACCORDING TO THE FEEDBACK. DO NOT REVISE MATERIAL WITHOUT EXPLICIT FEEDBACK.

--- HOST CONFIGURATION ---
**Host Personas:**
- Host A: {HOST_A_PERSONA}
- Host B: {HOST_B_PERSONA}

--- CUSTOM INSTRUCTIONS ---
{CUSTOM_INSTRUCTIONS}"""
        
        variable_suffix = f"""

--- INDEXED SCRIPT ---
{indexed_script}

--- REVISION FEEDBACK ---
{revision_feedback}"""
        
        full_prompt = static_base + variable_suffix
        
        print(f"  → Performing minor revision (line-level edits)...")
        
        start_time = time.time()
        if cache_name.startswith("files/"):
            file_obj = self.client.files.get(name=cache_name)
            response = self.client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=[file_obj, full_prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ScriptLineEdits,
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET_EDITOR)
                )
            )
        else:
            response = self.client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=[full_prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ScriptLineEdits,
                    cached_content=cache_name,
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET_EDITOR)
                )
            )
        
        execution_time = time.time() - start_time
        
        edits = response.parsed
        revised_script_with_keys, validation_errors = self._apply_line_edits(indexed_script, edits)
        
        if validation_errors:
            print(f"  ⚠️ Host prefix validation errors detected:")
            for error in validation_errors:
                print(f"    - {error}")
            print("  → Regenerating with host preservation warning...")
            
            enhanced_prompt = full_prompt + "\n\nIMPORTANT: You MUST NOT change any 'Host A:' or 'Host B:' prefixes. Only modify the content after the colon."
            
            start_time = time.time()
            if cache_name.startswith("files/"):
                file_obj = self.client.files.get(name=cache_name)
                response = self.client.models.generate_content(
                    model=DEFAULT_MODEL,
                    contents=[file_obj, enhanced_prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ScriptLineEdits,
                        thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET_EDITOR)
                    )
                )
            else:
                response = self.client.models.generate_content(
                    model=DEFAULT_MODEL,
                    contents=[enhanced_prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ScriptLineEdits,
                        cached_content=cache_name,
                        thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET_EDITOR)
                    )
                )
            
            execution_time = time.time() - start_time
            
            edits = response.parsed
            revised_script_with_keys, validation_errors = self._apply_line_edits(indexed_script, edits)
            
            if validation_errors:
                print(f"  ⚠️ Still has validation errors after regeneration - proceeding without problematic edits:")
                for error in validation_errors:
                    print(f"    - {error}")
        
        revised_script = self._strip_order_keys(revised_script_with_keys)
        
        if self.prompt_logger:
            self.prompt_logger.log_interaction(
                stage="script_revision_minor",
                prompt=full_prompt,
                response=response,
                metadata={
                    "original_script_length": len(script),
                    "revised_script_length": len(revised_script),
                    "revision_feedback": revision_feedback,
                    "edits_applied": [
                        {
                            "line_number": edit.line_number,
                            "content": edit.content
                        } for edit in edits.edits
                    ]
                }
            )
            
            self.prompt_logger.log_debug_text("script_revision_minor", indexed_script, "indexed_before")
            self.prompt_logger.log_debug_text("script_revision_minor", revised_script, "final_clean")
        
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            token_usage = TokenUsage(
                prompt_tokens=usage.prompt_token_count,
                cached_tokens=getattr(usage, 'cached_content_token_count', 0) or 0,
                candidate_tokens=usage.candidates_token_count,
                total_tokens=usage.total_token_count
            )
            self.logger.log_stage_metrics("script_revision_minor", token_usage, DEFAULT_MODEL, execution_time=execution_time)
        
        return revised_script
    
    def extract_claims(self, script: str) -> ExtractedClaims:
        """
        Extract factual claims from the script for verification.
        
        Args:
            script: The script to analyze
            
        Returns:
            ExtractedClaims with factual claims found
        """
        prompt = f"""Extract factual claims from the podcast script that should be verified against the source article.

--- CLAIMS TO EXTRACT ---
Focus on specific, verifiable statements that could be fact-checked:
- Research findings and key results
- Statistical data and specific numbers
- Methodological details about procedures
- Study conclusions and clinical implications
- Technical specifications or measurements

--- CLAIMS TO SKIP ---
Do NOT extract:
- General background information
- Common medical knowledge
- Obvious or widely-known facts
- Subjective opinions or interpretations

--- SCRIPT TO ANALYZE ---
{script}

--- FOCUS AREAS ---
Focus on specific, verifiable statements about research findings, statistics, methodologies, and conclusions."""
        
        start_time = time.time()
        response = self.client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractedClaims,
                thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
            )
        )
        execution_time = time.time() - start_time
        
        if self.prompt_logger:
            self.prompt_logger.log_interaction(
                stage="claim_extraction",
                prompt=prompt,
                response=response,
                metadata={"script_length": len(script)}
            )
        
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            token_usage = TokenUsage(
                prompt_tokens=usage.prompt_token_count,
                cached_tokens=0,
                candidate_tokens=usage.candidates_token_count,
                total_tokens=usage.total_token_count
            )
            self.logger.log_stage_metrics("claim_extraction", token_usage, DEFAULT_MODEL, execution_time=execution_time)
        
        return response.parsed
    
    def verify_claims(
        self, 
        claims: ExtractedClaims, 
        cache_name: str
    ) -> VerifiedClaimsAssessment:
        """
        Verify extracted claims against the source document.
        
        Args:
            claims: Claims to verify
            pdf_source: PDF file path or cache name depending on use_explicit_caching
            use_explicit_caching: If True, use explicit caching; else use implicit caching
            
        Returns:
            VerifiedClaimsAssessment with verification results
        """
        static_base = """Given a list of factual claims and source article text, verify each claim.
For each claim, first provide the reasoning by comparing the claim to the source material.
**CAVEAT** Your OCR ability is not perfect. When the script and source disagree with a technical word, use COMMON SENSE to figure out what is correct. **IMPORTANT**
Based ONLY on the reasoning you provide, conclude whether the claim's status is "supported", "unsupported", or "partially supported"."""
        
        claims_text = "\n".join([
            f"{i+1}. {claim.claim_text} (Category: {claim.claim_category})"
            for i, claim in enumerate(claims.claims)
        ])
        
        variable_suffix = f"""

--- CLAIMS TO VERIFY ---
{claims_text}"""
        
        full_prompt = static_base + variable_suffix
        
        start_time = time.time()
        if cache_name.startswith("files/"):
            file_obj = self.client.files.get(name=cache_name)
            response = self.client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=[file_obj, full_prompt],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=VerifiedClaimsAssessment,
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
                )
            )
        else:
            response = self.client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=[full_prompt],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=VerifiedClaimsAssessment,
                    cached_content=cache_name,
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
                )
            )
        execution_time = time.time() - start_time
        
        if self.prompt_logger:
            self.prompt_logger.log_interaction(
                stage="fact_verification",
                prompt=full_prompt,
                response=response,
                metadata={"claims_count": len(claims.claims)}
            )
        
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            token_usage = TokenUsage(
                prompt_tokens=usage.prompt_token_count,
                cached_tokens=getattr(usage, 'cached_content_token_count', 0) or 0,
                candidate_tokens=usage.candidates_token_count,
                total_tokens=usage.total_token_count
            )
            self.logger.log_stage_metrics("fact_verification", token_usage, DEFAULT_MODEL, execution_time=execution_time)
        
        return response.parsed
    
    def check_inappropriate_content(self, script: str) -> InappropriatenessCheck:
        """
        Check script for inappropriate content.

        Args:
            script: The script to check

        Returns:
            InappropriatenessCheck with assessment results
        """
        static_base = f"""Check if the podcast script contains any inappropriate content given the domain-specific guidance and host characteristics.
Also check for content that are potentially offensive, such as inappropriate remarks or nonverbal cues.

--- INAPPROPRIATE CONTENT EXAMPLES ---
- A male-voiced host talking about menstruation experience in a first-person perspective is inappropriate
- Offensive remarks, inappropriate jokes, or unprofessional commentary
- Content that violates the domain-specific guidelines below
- The hosts invite listeners to "contact us" or subscribe.

--- CUSTOM INSTRUCTIONS FOR SCRIPT GENERATION ---
{CUSTOM_INSTRUCTIONS}"""
        
        variable_suffix = f"""

--- SCRIPT TO CHECK ---
{script}"""
        
        full_prompt = static_base + variable_suffix
        
        start_time = time.time()
        response = self.client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=[full_prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=InappropriatenessCheck,
                thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
            )
        )
        execution_time = time.time() - start_time
        
        if self.prompt_logger:
            self.prompt_logger.log_interaction(
                stage="content_appropriateness",
                prompt=full_prompt,
                response=response,
                metadata={"script_length": len(script)}
            )
        
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            token_usage = TokenUsage(
                prompt_tokens=usage.prompt_token_count,
                cached_tokens=0,
                candidate_tokens=usage.candidates_token_count,
                total_tokens=usage.total_token_count
            )
            self.logger.log_stage_metrics("content_appropriateness", token_usage, DEFAULT_MODEL, execution_time=execution_time)
        
        return response.parsed
    
    def run_full_qa_cycle(
        self, 
        script: str, 
        cache_name: str,
        target_word_length: int = None,
        max_iterations: int = 2
    ) -> str:
        """
        Run the complete QA cycle with multiple attempts per step.
        
        Args:
            script: Initial script
            cache_name: Cache name for PDF access
            target_word_length: Target word length for major revisions (optional)
            max_iterations: Maximum number of attempts per step
            
        Returns:
            Final approved script
        """
        current_script = script
        original_script = script
        previous_feedback = None
        
        print("  → Running quality assessment...")
        judgment = self.judge_script(current_script, cache_name)
        print(f"  → Judge verdict: {judgment.verdict} (score: {judgment.score}/10)")
        
        if judgment.assessment_category == "Minor Revision":
            current_script = self.revise_script_minor(current_script, judgment.evaluation_feedback, cache_name)
            print("  ✓ Minor revision completed")
        elif judgment.assessment_category in ["Major Revision", "Reject"]:
            if target_word_length:
                print("  → Major revision required - generating new script with length correction...")
                script_generator = ScriptGenerator(self.client, self.logger, self.prompt_logger)
                
                enhanced_prompt = f"""REVISION FEEDBACK: {judgment.evaluation_feedback}
                
Please generate a new script that addresses all the feedback above while maintaining the target word length of {target_word_length} words."""
                
                result = script_generator.generate_script_with_length_correction(
                    cache_name, target_word_length, additional_instructions=enhanced_prompt
                )
                current_script = result.script_text
                print(f"  → New script generated: {result.word_count} words (target: {target_word_length})")
                
                print("  → Re-evaluating new script...")
                new_judgment = self.judge_script(current_script, cache_name)
                print(f"  → New judge verdict: {new_judgment.verdict} (score: {new_judgment.score}/10)")
                
                if new_judgment.assessment_category == "Minor Revision":
                    current_script = self.revise_script_minor(current_script, new_judgment.evaluation_feedback, cache_name)
                    print("  ✓ Minor revision applied to new script")
                elif new_judgment.assessment_category in ["Major Revision", "Reject"]:
                    print("  ⚠ New script still needs major revision - proceeding with current version")
                
                print("  ✓ Major revision (rewrite with length check) completed")
            else:
                current_script = self.rewrite_script_major(current_script, judgment.evaluation_feedback, cache_name)
                print("  ✓ Major revision (simple rewrite) completed")
        else:
            print("  ✓ Quality assessment passed")
        
        print("  → Running fact-checking...")
        claims = self.extract_claims(current_script)
        
        if claims.claims:
            print(f"  → Found {len(claims.claims)} factual claims to verify")
            verification = self.verify_claims(claims, cache_name)
            print(f"  → Fact-check accuracy: {verification.overall_accuracy_score:.1%}")
            
            if verification.overall_accuracy_score >= 1.0:
                print("  ✓ Fact-checking passed")
            else:
                print(f"  → Applying fact-check revision...")
                
                failed_claims = []
                for claim in verification.verified_claims:
                    if claim.verification_status in ["unsupported", "partially supported"]:
                        failed_claims.append(f"• '{claim.claim_text}' - {claim.verification_status}: {claim.verification_reasoning}")
                
                if failed_claims:
                    feedback = f"Fix these factual inaccuracies by replacing the incorrect information:\n" + "\n".join(failed_claims)
                    current_script = self.revise_script_minor(current_script, feedback, cache_name)
                    print("  ✓ Fact-checking revision completed")
                else:
                    print("  → No specific factual claims to fix")
        else:
            print("  → No factual claims found to verify")
        
        print("  → Running content appropriateness check...")
        for attempt in range(max_iterations):
            print(f"  → Content check attempt {attempt + 1}...")
            appropriateness = self.check_inappropriate_content(current_script)
            
            if appropriateness.is_appropriate:
                print("  ✓ Content appropriateness check passed")
                break
            elif attempt < max_iterations - 1:  # Don't revise on final attempt
                print(f"  → Applying content appropriateness revision...")
                
                if appropriateness.flagged_sections:
                    flagged_text = "\n".join([f"• {section}" for section in appropriateness.flagged_sections])
                    feedback = f"Fix these inappropriate content issues:\n{flagged_text}\n\nGeneral guidance: {appropriateness.feedback}"
                else:
                    feedback = f"Fix content appropriateness issues: {appropriateness.feedback}"
                    
                current_script = self.revise_script_minor(current_script, feedback, cache_name)
            else:
                print(f"  → Content appropriateness completed after {max_iterations} attempts")
        
        print("  ✓ Quality assurance completed")
        return current_script