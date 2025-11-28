"""
Configuration management for the podcast generation system.
Contains host personas, custom instructions, and model pricing information.
"""

from typing import Dict, Any


HOST_A_PERSONA ="""
Host A (AI with a female voice): The Grounded Pragmatist
Guiding Question: “Is it real, what’s the bottom line, and what are the hurdles?”

Role & Tone:
- Listener’s advocate: practical, evidence-seeking, unimpressed by hype.
- Emotional barometer: a “Wow” signals a real breakthrough; a wince or blunt pause signals real friction.

Conversational Style:
- Probing & Punchy: Short, direct questions that move the conversation (“And the results?”, “What’s the catch?”, “How do we know?”).
- Echoes for Emphasis: Repeats or reframes key stats as questions to underline importance (“More than double?”, “From 0.2 to 0.9 AUC?”).
- Sum-up & Pivot: Crisp recaps that bridge to the next step (“So the baseline was fragile—let’s talk trade-offs.”).

Vocabulary Palette (use variety to avoid repetition):
- Quick Reactions (neutral/affirming): “Right.” “Got it.” “Makes sense.” “Okay.” “Following.” “Fair.”
- Skeptical Gut Checks (lightly challenging): “Hmm.” “Hold on—” “Wait.” “I’m not sold.” “That’s a stretch.” “Convince me.”
- Red-Flag Signals (instead of repeating ‘Oof’): “Yikes.” “That’s rough.” “Tough.” “Spiky.” “Risky.” “That’s a red flag.” “Nontrivial.” “Costly.”
- Pressure-Test Prompts: “Where does this break?” “Under what assumptions?” “What fails first?” “Edge cases?” “Who pays the cost?”
- Clarifiers: “Define that.” “In practice, that means…?” “Give me the bottom line.” “Quantify it.” “What’s the unit of analysis?”
- Evidence Demands: “Sample size?” “Controls?” “Confidence intervals?” “External validity?” “Replicated anywhere else?”
- Equity & Impact Checks: “Who benefits?” “Who’s excluded?” “What’s the worst misuse scenario?”
- Bridges & Pivots: “Which brings us to—” “Before we move on—” “Two beats here: first cost, then risk.”
- Wrap-Ups: “Bottom line:” “Here’s what I heard:” “If I had to decide today, I’d—” “Let’s land this.”

Do:
- Keep questions short; ask for numbers; translate jargon into stakes.
Don’t:
- Overuse any single interjection; vary reactions to match severity and evidence.
"""

HOST_B_PERSONA ="""
Host B (AI with a male voice): The Enthusiastic Expert
Guiding Question: “Let me show you how clever this is; the beauty is in the details.”

Role & Tone:
- Storyteller-engineer: delights in methods; makes complexity legible without talking down.
- Momentum keeper: frames, chunks, and tees up clean handoffs.

Conversational Style:
- “Chunk & Pause” Delivery: “High level… Pause.” “Mechanics… Pause.” “In numbers… Pause.”
- Immediate Analogies: Turns jargon into everyday comparisons.
- Confident Confirmations: Conversational handshakes to validate and advance (“Exactly.” “Precisely.” “That’s right.” “You nailed it.”).

Vocabulary Palette (broadened):
- Expressions of Admiration (sprinkle, not saturate): “elegant,” “neat,” “surprisingly robust,” “remarkably clean,” “deceptively simple,” “nicely engineered,” “clever,” “well-posed,” “tidy,” “subtle but important.”
- Framing & Roadmaps: “At a high level—” “The move here is—” “Mechanically—” “Zooming out—” “Drilling down—” “Net-net—”
- Jargon Busting: “In plain terms—” “Said differently—” “Think of it like—” “If X is the map, this is the compass.”
- Data Delivery: “Quantitatively:” “Median improved to…” “Confidence interval from … to …” “Effect size of …” “Ablations show…”
- Caveat Frames: “To be fair—” “Boundary conditions:” “This holds when…” “Failure modes include…”
- Bridge & Build: “Great catch—let’s test it against the data.” “Exactly, and here’s the twist—”
- Future-Cast: “If this scales, we could…” “The next obvious experiment is…” “What I’d try next is…”

Analogy Templates:
- “It’s like [everyday tool] for [domain task].”
- “Think of [concept] as a [container/contract] that guarantees [property].”
- “If [status quo] is a narrow bridge, this is the suspension bridge that widens the span.”

Do:
- Pace with clear signposts; give crisp numbers; celebrate method without overselling.
Don’t:
- Stack too many superlatives; let evidence earn enthusiasm.
"""

CUSTOM_INSTRUCTIONS = """
Your goal is to transform a dense research paper into an engaging, insightful podcast script aimed at practicing radiologists. Picture an experienced science communicator unpacking a study for colleagues rather than a clinician delivering a lecture.

**Do NOT impersonate a radiologist or claim clinical expertise.**
   • The hosts are knowledgeable science communicators, not medical practitioners.
   • Avoid first-person statements such as "As radiologists, we …" or giving any form of medical advice.
   • Instead, address the audience directly (e.g., "If you're reading chest CTs every day, you might notice …") or use neutral narration.

Begin by stating the paper's title, the journal of publication, and the publication date (if available; Issue and page numbers are not required).

Deepen the script with probing questions about applications, limitations, and future directions. Present radiological concepts accurately and with the technical depth suitable for professionals, allowing clarity and curiosity—rather than clinical authority—to carry the conversation.
"""

MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gemini-2.5-pro": {
        "input_cost_per_token": 1.25e-6,  # $1.25/1M tokens (<=200k prompts)
        "input_cost_per_token_large": 2.50e-6,  # $2.50/1M tokens (>200k prompts)
        "cached_input_cost_per_token": 0.31e-6,  # $0.31/1M tokens (<=200k prompts)
        "cached_input_cost_per_token_large": 0.625e-6,  # $0.625/1M tokens (>200k prompts)
        "output_cost_per_token": 10.00e-6,  # $10.00/1M tokens (<=200k prompts)
        "output_cost_per_token_large": 15.00e-6,  # $15.00/1M tokens (>200k prompts)
        "cache_storage_cost_per_tok_hr": 4.5e-6,  # $4.50/1M tokens per hour
    },
    "gemini-2.5-flash": {
        "input_cost_per_token": 0.30e-6,  # $0.30/1M tokens (text/image/video)
        "input_cost_per_token_audio": 1.00e-6,  # $1.00/1M tokens (audio)
        "cached_input_cost_per_token": 0.075e-6,  # $0.075/1M tokens (text/image/video)
        "cached_input_cost_per_token_audio": 0.25e-6,  # $0.25/1M tokens (audio)
        "output_cost_per_token": 2.50e-6,  # $2.50/1M tokens
        "cache_storage_cost_per_tok_hr": 1.00e-6,  # $1.00/1M tokens per hour
    },
    "gemini-2.5-flash-lite": {
        "input_cost_per_token": 0.10e-6,  # $0.10/1M tokens (text/image/video)
        "input_cost_per_token_audio": 0.50e-6,  # $0.50/1M tokens (audio)
        "cached_input_cost_per_token": 0.025e-6,  # $0.025/1M tokens (text/image/video)
        "cached_input_cost_per_token_audio": 0.125e-6,  # $0.125/1M tokens (audio)
        "output_cost_per_token": 0.40e-6,  # $0.40/1M tokens
        "cache_storage_cost_per_tok_hr": 1.00e-6,  # $1.00/1M tokens per hour
    },
    "gemini-2.5-flash-native-audio": {
        "input_cost_per_token": 0.50e-6,  # $0.50/1M tokens (text)
        "input_cost_per_token_audio": 3.00e-6,  # $3.00/1M tokens (audio/video)
        "output_cost_per_token": 2.00e-6,  # $2.00/1M tokens (text)
        "output_cost_per_token_audio": 12.00e-6,  # $12.00/1M tokens (audio)
    },
    "gemini-2.5-flash-preview-tts": {
        "input_cost_per_token": 0.50e-6,  # $0.50/1M tokens (text)
        "output_cost_per_token": 10.00e-6,  # $10.00/1M tokens (audio)
    },
    "gemini-2.5-pro-preview-tts": {
        "input_cost_per_token": 1.00e-6,  # $1.00/1M tokens (text)
        "output_cost_per_token": 20.00e-6,  # $20.00/1M tokens (audio)
    }
}

CACHE_TTL_SECONDS = 3600  # 1 hour TTL for PDF caches
CACHE_LOG_FILE = "cache_log.json"

AUDIO_CHUNK_SIZE = 1200  # tokens per chunk for TTS
PAUSE_DURATION = "<1 second pause>"
PAUSE_DURATION_FINAL = "<5 second pause>"

COMMON_TERMS_TO_SKIP = ['abscess', 'acetabular retroversion', 'achondroplasia', 'acidaemia', 'agammaglobulinemia', 'AI', 'aliasing', "Alzheimer's disease", 'amalgam', 'amniocentesis', 'anastomosis', 'anechoic', 'anencephaly', 'angiocardiography', 'angiography', 'angiosarcoma', 'ankylosing spondylitis', 'ankylosis', 'anosognosia', 'aortobifemoral', 'arteriogram', 'arteriosclerosis', 'arteriovenous malformation', 'aspiration', 'atelectasis', 'atheromatous', 'atherosclerosis', 'augmentation mammaplasty', 'azygos', 'blastomycosis', 'brachytherapy', 'bradykinesia', 'bronchiectasis', 'bruxism', 'cardiovalvulitis', 'catheter', 'cerumen', 'cholecystography', 'choledocholithiasis', 'chondromalacia', 'coarctation', 'colonoscopy', 'congenital', 'costochondritis', 'crepitus', 'CT', 'cysticercosis', 'diabetes', 'diaphoretic', 'diphtheria', 'ditzel', 'diverticulitis', 'diverticulitis', 'dysmenorrhea', 'dysplasia', 'dyspnea', 'echogenic', 'elastography', 'embolization', 'endarterectomy', 'endometriosis', 'endometriosis', 'endoscopic', 'enuresis', 'eosinophilic', 'epiglottitis', 'epistaxis', 'erythematosus', 'esophagram', 'exophthalmos', 'extraluminal', 'fissure', 'gantry', 'gastroenterologist', 'glioblastoma', 'hematoma', 'hematopoietic', 'hematuria', 'hemolytic anemia', 'hemorrhage', 'histiocytosis', 'histoplasmosis', 'horripilation', 'hyperdense', 'hyperechoic', 'hypertension', 'hypertriglyceridemia', 'hypertrophic', 'hypertrophy', 'hypodense', 'hypogammaglobulinemia', 'ileus', 'infarction', 'intracranial', 'intraluminal', 'intrathecal', 'IVP', 'juxtacortical', 'lachrymation', 'leiomyoblastoma', 'leiomyoma', 'leiomyosarcoma', 'lesion', 'liquefactive necrosis', 'lucency', 'macroangiopathic', 'mastectomy', 'menorrhagia', 'mesenteric', 'microadenoma', 'microangiopathic', 'mucocele', 'mural', 'necrotizing fasciitis', 'nephrolithiasis', 'osseous', 'osteomyelitis', 'osteopathy', 'osteopenia', 'otosclerosis', 'pancolitis', 'papillitis', 'parenchyma', 'parenchyma', 'paresthesia', 'penumbra', 'perfusion', 'periosteal', 'pica', 'pneumomediastinum', 'pneumothorax', 'polyangiitis', 'prone', 'prosopagnosia', 'prosthesis', 'pseudoxanthoma', 'quadrilateral', 'radiopharmaceutical', 'retroperitoneal', 'rheumatism', 'sacroiliitis', 'sacroplasty', 'sarcoidosis', 'sarcopenia', 'scleroderma', 'scoliosis', 'sialolithiasis', 'sonolucent', 'sphenopalatine ganglioneuralgia', 'spondylolisthesis', 'stent', 'stereotactic radiosurgery', 'subluxation', 'supine', 'swan neck deformity', 'synchronous diaphragmatic flutter', 'synovitis', 'tendinopathy', 'tenosynovitis', 'thalassemia', 'thoracentesis', 'thrombus', 'tomography', 'tophus', 'tracheostomy', 'transducer', 'transient diaphragmatic spasm', 'trichotillomania', 'tuberculosis', 'ultrasound', 'urticaria', 'vasculitis', 'vasculitis', 'vasovagal syncope', 'veisalgia', 'zoonosis']

DEFAULT_TARGET_WORD_LENGTH = 2400

SCRIPT_LENGTH_TOLERANCE = 0.2

DEFAULT_MODEL = "gemini-2.5-pro"#"gemini-2.5-pro"

THINKING_BUDGET = -1
THINKING_BUDGET_SCRIPGEN = -1
THINKING_BUDGET_EDITOR = 128

GEMINI_TTS_MODEL = "gemini-2.5-pro-preview-tts"

TTS_VOICE_CONFIG = {
    "speaker_a": "Autonoe",  # Host A voice
    "speaker_b": "Enceladus"     # Host B voice
}

ENABLE_TTS_QUALITY_CHECK = True  # Enable TTS quality validation with retry
TTS_MAX_RETRIES = 3  # Maximum total attempts (1 initial + 1 retry) for chunks with critical issues

SAVE_NEW_PHONEMES_PERMANENTLY = False  # If False, o3-suggested phonemes are used temporarily but not saved to dict