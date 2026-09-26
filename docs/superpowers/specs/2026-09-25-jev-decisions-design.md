# Jev Decisions Integration Design

## Goal

Add active, visible decision signals to POZZ's simulation evaluation and recorded-interview draft workflows without changing clinical content, triage, prescriptions, or the clinician's authority.

## Boundary

The FastAPI backend calls OpenRouter's Decisions API with `OPENROUTER_API_KEY`, endpoint `https://openrouter.ai/api/alpha/decisions`, and pinned model `typesafe/jev-1.13`. The state contains only the minimal conversation or draft excerpt needed for the question. Questions and criteria are English even when the evaluated material is Polish.

## Flow

1. After the existing LLM evaluation is produced, evaluate `training_feedback_completeness` (score), `draft_has_clinician_review_boundary` (noul), and `transcript_attribution_needs_review` (noul).
2. Render results as labelled, non-clinical training signals in the existing evaluation/detail surfaces. An under-threshold or failed Jev request emits no new signal and never blocks evaluation completion.
3. For each high-probability missing training dimension, show only a neutral review cue; Jev cannot generate suggested diagnosis, medication, investigation, or treatment.
4. Existing hidden gold-plan evaluation, human review, RLS, spend cap, and draft-required warnings remain unchanged and authoritative.

## Routing and failure handling

Retry transient 429/5xx responses with bounded exponential backoff. On any exhausted retry, invalid response, or answer below the conservative threshold, record `unavailable_or_uncertain` and continue the existing flow. Jev cannot gate a patient simulation, clinical draft, or recorded interview.

## Observability and privacy

Store an audit event with question IDs, answer values/probabilities, model/provider/request ID, token counts, latency, selected display route, and eventual clinician correction where the UI offers one. Store a digest of the minimal state rather than a second transcript copy. Existing transcript retention, ownership checks, and RLS apply unchanged.

## Tests

Use mocked Decisions API fixtures to test complete, uncertain, and unavailable outcomes; assert no outcome changes a clinical draft or bypasses its clinician-review label. Add a regression test proving raw transcript text is not duplicated into decision-event storage.

## Non-goals

Jev does not diagnose, triage, prescribe, select investigations, evaluate real patients for clinical use, or replace the existing LLM evaluation. The sole runtime exception is the non-clinical `voice_turn_complete` boundary check specified in [`2026-09-26-jev-voice-turns-design.md`](2026-09-26-jev-voice-turns-design.md); its failure rejects the transient voice turn rather than changing clinical content.
