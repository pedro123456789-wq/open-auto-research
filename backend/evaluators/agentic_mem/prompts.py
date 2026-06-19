"""
Official LoCoMo prompts.

Two sets of prompts live here:
  QA prompts  — used to ask the agent/LLM to answer a LoCoMo question.
                Mirrors the prompts in task_eval/gpt_utils.py.
  Judge prompts — used by an LLM judge to score a pre-generated answer.
                  Mirrors the rubric used by memory-benchmark / mem0.
"""

# Used for all categories except 5 (adversarial).
QA_PROMPT = (
    "Based on the above context, write an answer in the form of a short phrase "
    "for the following question. Answer with exact words from the context whenever possible.\n\n"
    "Question: {question} Short answer:"
)

# Used for category 5 (adversarial) — no leading instruction to use exact words,
# because the expected answer is "Not mentioned in the conversation".
QA_PROMPT_CAT_5 = (
    "Based on the above context, answer the following question.\n\n"
    "Question: {question} Short answer:"
)

# Prepended to every conversation context fed to the model.
CONV_START_PROMPT = (
    "Below is a conversation between two people: {speaker_a} and {speaker_b}. "
    "The conversation takes place over multiple days and the date of each "
    "conversation is written at the beginning of the conversation.\n\n"
)

# Template for a single conversation session block inside the context.
SESSION_BLOCK = "DATE: {date}\nCONVERSATION:\n{turns}\n\n"


def build_qa_prompt(question: str, category: int) -> str:
    """Return the appropriate QA prompt string for the given category."""
    template = QA_PROMPT_CAT_5 if category == 5 else QA_PROMPT
    return template.format(question=question)


def build_conv_start(speaker_a: str, speaker_b: str) -> str:
    """Return the conversation-start header for the given speaker names."""
    return CONV_START_PROMPT.format(speaker_a=speaker_a, speaker_b=speaker_b)


# -------------------------------------------------------------------------------
# Judge prompts  (LLM-as-judge)
# -------------------------------------------------------------------------------
JUDGE_SYSTEM = (
    "You are evaluating conversational AI memory recall. "
    "Return JSON only with the format requested."
)

JUDGE_PROMPT = """Label the generated answer as CORRECT or WRONG.

## Rules
1. PARTIAL CREDIT: if the generated answer contains at least one correct item
   from the gold answer, mark CORRECT.
2. PARAPHRASES COUNT: the same concept in different words is CORRECT.
3. EXTRA DETAIL IS FINE: a longer answer that still contains the gold answer's
   key facts is CORRECT.
4. DATE TOLERANCE: dates within ~2 weeks, or durations within 50%, are CORRECT.
5. MULTI-ANSWER: the gold answer may list several items separated by commas —
   containing at least one of them is enough to mark CORRECT.
6. Mark WRONG only if the generated answer shares no correct item with the gold
   answer or addresses a completely different topic.
7. If the answer contains a lot of superfluous details, and is not coherent mark as wrong, 
   even if it contains the correct answer.

## Question
Question: {question}
Gold answer: {answer}
Generated answer: {response}

Return JSON with "reasoning" (one sentence) and "label" (CORRECT or WRONG)."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "label": {"type": "string", "enum": ["CORRECT", "WRONG"]},
    },
    "required": ["reasoning", "label"],
    "additionalProperties": False,
}
