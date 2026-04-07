"""
Pydantic models for structured outputs from the Gemini API.
These models ensure reliable, parsable results from LLM calls.
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from datetime import datetime


class GeneratedScript(BaseModel):
    """Structured output for LLM-generated podcast scripts."""
    script_text: str = Field(description="Complete podcast script with exact speaker prefixes 'Host A: ' and 'Host B: ' (with space after colon) for each turn, separated by newlines. No preamble, commentary, or text outside the script dialogue.")

class ScriptOutput(BaseModel):
    """Output model for generated podcast scripts with metadata."""
    script_text: str = Field(description="The complete podcast script with Host A: and Host B: prefixes")
    word_count: int = Field(description="Total word count of the script")
    estimated_duration_minutes: float = Field(description="Estimated duration in minutes")


class JudgeOutput(BaseModel):
    """Output model for script quality assessment."""
    evaluation_feedback: str = Field(description="Detailed feedback on the script quality")
    assessment_category: Literal["Accept", "Minor Revision", "Major Revision", "Reject"] = Field(
        description="Accept: Flawless or trivial corrections only (use sparingly). Minor Revision: Issues that can be fixed at word/phrase/line level without adding or removing content (most common). Major Revision: Requires overall re-do, addition/deletion of lines, or structural changes. Reject: Fundamentally flawed, requires complete rewrite (use sparingly)."
    )
    verdict: str = Field(description="Overall verdict (same as assessment_category for compatibility)")
    score: int = Field(ge=1, le=10, description="Quality score from 1 to 10")


class ScriptLineEdit(BaseModel):
    """Individual line edit operation for minor script revisions."""
    line_number: int = Field(
        description="Line number to replace (e.g., 1, 2, 3). Cannot add new lines."
    )
    content: str = Field(
        description="New content for the line. Include the complete line with speaker prefix (e.g., 'Host A: So fine-tuning was key...')."
    )


class ScriptLineEdits(BaseModel):
    """Collection of line-based edits for script revision."""
    edits: List[ScriptLineEdit] = Field(
        default_factory=list,
        description="A comprehensive list of all line-based edits to be applied"
    )


class FactualClaim(BaseModel):
    """Individual factual claim extracted from script."""
    claim_text: str = Field(description="The specific factual claim made in the script")
    claim_category: str = Field(description="Category of the claim (e.g., statistical, methodological, clinical)")
    source_line_numbers: List[int] = Field(description="Line numbers where this claim appears")


class ExtractedClaims(BaseModel):
    """Collection of factual claims extracted from script."""
    claims: List[FactualClaim] = Field(description="List of factual claims found in the script")


class VerifiedClaim(BaseModel):
    """Verification result for a single factual claim."""
    claim_text: str = Field(description="The original claim being verified")
    verification_reasoning: str = Field(description="Detailed reasoning for the verification")
    verification_status: Literal["supported", "unsupported", "partially supported"] = Field(
        description="Verification result based on source document"
    )
    source_references: List[str] = Field(
        description="Specific references from source document that support or contradict the claim"
    )


class VerifiedClaimsAssessment(BaseModel):
    """Complete fact-checking assessment of all claims."""
    verified_claims: List[VerifiedClaim] = Field(description="List of all verified claims")
    overall_accuracy_score: float = Field(
        ge=0.0, le=1.0,
        description="Overall accuracy score from 0.0 to 1.0"
    )
    accuracy_summary: str = Field(description="Summary of the overall accuracy assessment")


class InappropriatenessCheck(BaseModel):
    """Assessment of content appropriateness."""
    is_appropriate: bool = Field(description="Whether the content is appropriate")
    feedback: str = Field(description="Detailed feedback on appropriateness issues, if any")
    flagged_sections: List[str] = Field(
        default_factory=list,
        description="Specific sections or lines that were flagged as inappropriate"
    )


class TermIPA(BaseModel):
    """Individual term with its IPA pronunciation."""
    term: str = Field(description="The exact term as it appears in the script")
    ipa: str = Field(description="IPA pronunciation guide for the term")


class IdentifiedTerms(BaseModel):
    """Collection of terms requiring phoneme tags."""
    terms: List[TermIPA] = Field(description="List of terms that need IPA phoneme tags")


class AudioChunk(BaseModel):
    """Individual audio chunk information."""
    chunk_id: int = Field(description="Sequential ID of the chunk")
    text_content: str = Field(description="Text content for this chunk")
    speaker_starts: List[str] = Field(description="List of speakers that start in this chunk")
    word_count: int = Field(description="Word count of this chunk")
    human_readable_text: Optional[str] = Field(default=None, description="Human-readable version without phoneme tags")


class AudioGeneration(BaseModel):
    """Complete audio generation configuration."""
    chunks: List[AudioChunk] = Field(description="List of audio chunks to generate")
    total_chunks: int = Field(description="Total number of chunks")
    estimated_duration_minutes: float = Field(description="Estimated total duration")


class TokenUsage(BaseModel):
    """Token usage tracking for API calls."""
    prompt_tokens: int = Field(description="Number of prompt tokens used")
    cached_tokens: int = Field(default=0, description="Number of cached tokens used") 
    candidate_tokens: int = Field(description="Number of output tokens generated")
    total_tokens: int = Field(description="Total tokens used")


class StageMetrics(BaseModel):
    """Metrics for a single pipeline stage."""
    stage_name: str = Field(description="Name of the pipeline stage")
    execution_time_seconds: float = Field(description="Time taken to execute this stage")
    token_usage: TokenUsage = Field(description="Token usage for this stage")
    cost_usd: float = Field(description="Estimated cost in USD for this stage")
    model_used: str = Field(description="Model used for this stage")
    cache_hit: bool = Field(default=False, description="Whether cache was used")


class PipelineLog(BaseModel):
    """Complete pipeline execution log."""
    pipeline_id: str = Field(description="Unique identifier for this pipeline run")
    source_pdf: str = Field(description="Source PDF filename")
    cache_name: str = Field(description="Cache name used for this run")
    start_time: datetime = Field(description="Pipeline start timestamp")
    end_time: Optional[datetime] = Field(default=None, description="Pipeline end timestamp")
    total_execution_time_seconds: Optional[float] = Field(default=None, description="Total execution time")
    stage_metrics: List[StageMetrics] = Field(description="Metrics for each pipeline stage")
    total_cost_usd: float = Field(description="Total estimated cost in USD")
    status: Literal["running", "completed", "failed"] = Field(description="Pipeline execution status")
    error_message: Optional[str] = Field(default=None, description="Error message if pipeline failed")


class TTSIssue(BaseModel):
    """Individual TTS quality issue identified in audio."""
    issue_type: Literal["missing_content", "mispronunciation", "wrong_speaker", "audio_corruption", "timing_issue"] = Field(
        description="Type of TTS issue detected"
    )
    severity: Literal["critical", "minor"] = Field(
        description="Severity level: critical issues require retry, minor issues are logged only"
    )
    description: str = Field(description="Detailed description of the issue")
    timestamp_start: Optional[float] = Field(default=None, description="Start time in seconds where issue occurs")
    timestamp_end: Optional[float] = Field(default=None, description="End time in seconds where issue occurs")
    expected_text: Optional[str] = Field(default=None, description="Text that should have been spoken")
    actual_text: Optional[str] = Field(default=None, description="Text that was actually spoken (if detectable)")


class TTSQualityAssessment(BaseModel):
    """Complete TTS quality assessment for an audio chunk."""
    issues: List[TTSIssue] = Field(description="List of all issues found in the audio")
    has_critical_issues: bool = Field(description="Whether critical issues requiring retry were found")
    overall_quality_score: float = Field(
        ge=0.0, le=1.0,
        description="Overall quality score from 0.0 (poor) to 1.0 (excellent)"
    )