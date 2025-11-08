import os
import uuid
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional, Tuple

import psycopg  # type: ignore

from .config import get_database_url


_GLOBAL_CONN: Optional[psycopg.Connection] = None


def _normalize_dsn(dsn: str) -> str:
    """Normalize connection URL to a psycopg-compatible DSN.

    Accepts SQLAlchemy-style URL like 'postgresql+psycopg://...'
    and converts it to 'postgresql://...'.
    """
    if dsn.startswith("postgresql+psycopg://"):
        return dsn.replace("postgresql+psycopg://", "postgresql://", 1)
    return dsn


def _get_conn() -> psycopg.Connection:
    """Return a global connection; create it on first use."""
    global _GLOBAL_CONN
    if _GLOBAL_CONN is None:
        dsn = get_database_url()
        if not dsn:
            from .config import get_environment
            env = get_environment()
            error_msg = (
                "DATABASE_URL not configured.\n"
                f"Environment: {env}\n"
            )
            if env == "local":
                error_msg += "Set DATABASE_URL in .env file or environment variables."
            else:
                secret_name = os.getenv("POSTGRES_SECRET_NAME", "POZZ")
                region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "eu-central-1"
                error_msg += (
                    f"Set DATABASE_URL in AWS Secrets Manager.\n"
                    f"Expected secret name: {secret_name}\n"
                    f"Region: {region}\n"
                    f"Or set DATABASE_URL environment variable.\n"
                    f"Check application logs for detailed error messages."
                )
            raise RuntimeError(error_msg)
        dsn = _normalize_dsn(dsn)
        _GLOBAL_CONN = psycopg.connect(dsn, autocommit=True)
    return _GLOBAL_CONN


@contextmanager
def get_cursor() -> Generator[psycopg.Cursor, None, None]:
    conn = _get_conn()
    with conn.cursor() as cur:
        yield cur


def init_schema() -> None:
    """Create tables if they do not exist."""
    with get_cursor() as cur:
        cur.execute(
            """
            create table if not exists patients (
                id uuid primary key,
                created_at timestamptz not null default now(),
                scenario text not null,
                summary text,
                treatment_plan text
            );
            """
        )
        # Add summary column if it doesn't exist (for existing DBs)
        try:
            cur.execute("alter table patients add column if not exists summary text")
        except Exception:
            pass  # Column might already exist
        # Add treatment_plan column if it doesn't exist (for existing DBs)
        try:
            cur.execute("alter table patients add column if not exists treatment_plan text")
        except Exception:
            pass  # Column might already exist
        cur.execute(
            """
            create table if not exists conversations (
                id uuid primary key,
                patient_id uuid not null references patients(id) on delete cascade,
                created_at timestamptz not null default now(),
                title text,
                user_treatment_response text,
                diagnosis_evaluation text
            );
            """
        )
        # Add new columns if they don't exist (for existing DBs)
        try:
            cur.execute("alter table conversations add column if not exists user_treatment_response text")
        except Exception:
            pass
        try:
            cur.execute("alter table conversations add column if not exists diagnosis_evaluation text")
        except Exception:
            pass
        # Add columns for interview recording
        try:
            cur.execute("alter table conversations add column if not exists interview_summary text")
        except Exception:
            pass
        try:
            cur.execute("alter table conversations add column if not exists extracted_info jsonb")
        except Exception:
            pass
        cur.execute(
            """
            create table if not exists messages (
                id bigserial primary key,
                conversation_id uuid not null references conversations(id) on delete cascade,
                role varchar(16) not null,
                content text not null,
                created_at timestamptz not null default now()
            );
            """
        )
        # Table for interview transcripts (with LLM-assigned roles)
        cur.execute(
            """
            create table if not exists interview_transcripts (
                id uuid primary key,
                conversation_id uuid not null references conversations(id) on delete cascade,
                chunk_number int not null,
                transcript_json jsonb not null,
                created_at timestamptz not null default now()
            );
            """
        )
        # Table for LLM suggestions during recording (every minute)
        cur.execute(
            """
            create table if not exists interview_suggestions (
                id uuid primary key,
                conversation_id uuid not null references conversations(id) on delete cascade,
                chunk_number int not null,
                minute_number int not null,
                suggestions text not null,
                created_at timestamptz not null default now()
            );
            """
        )


def create_patient(scenario: str, summary: Optional[str] = None, treatment_plan: Optional[str] = None) -> str:
    """Create a new patient record with scenario, summary, and treatment plan."""
    patient_id = str(uuid.uuid4())
    with get_cursor() as cur:
        cur.execute(
            "insert into patients (id, scenario, summary, treatment_plan) values (%s, %s, %s, %s)",
            (patient_id, scenario, summary, treatment_plan),
        )
    return patient_id


def get_patient_treatment_plan(patient_id: str) -> Optional[str]:
    """Get treatment plan for a patient."""
    with get_cursor() as cur:
        cur.execute("select treatment_plan from patients where id = %s", (patient_id,))
        row = cur.fetchone()
        return row[0] if row and row[0] else None


def update_patient_summary(patient_id: str, summary: str) -> None:
    """Update summary for existing patient."""
    with get_cursor() as cur:
        cur.execute("update patients set summary = %s where id = %s", (summary, patient_id))


def create_conversation(patient_id: str, title: Optional[str] = None) -> str:
    conversation_id = str(uuid.uuid4())
    with get_cursor() as cur:
        cur.execute(
            "insert into conversations (id, patient_id, title) values (%s, %s, %s)",
            (conversation_id, patient_id, title),
        )
    return conversation_id


def add_message(conversation_id: str, role: str, content: str) -> None:
    with get_cursor() as cur:
        cur.execute(
            "insert into messages (conversation_id, role, content) values (%s, %s, %s)",
            (conversation_id, role, content),
        )


def get_conversation_messages(conversation_id: str) -> List[Tuple[int, str, str, str]]:
    """Return list of (id, role, content, created_at ISO)."""
    with get_cursor() as cur:
        cur.execute(
            "select id, role, content, to_char(created_at, 'YYYY-MM-DD" "HH24:MI:SSOF') as created_at from messages where conversation_id=%s order by id asc",
            (conversation_id,),
        )
        rows = cur.fetchall()
    return [(r[0], r[1], r[2], r[3]) for r in rows]


def get_conversation_details(conversation_id: str) -> Optional[Tuple[str, str, Optional[str], Optional[str]]]:
    """Get conversation details including user response and evaluation.
    
    Returns: (patient_id, title, user_treatment_response, diagnosis_evaluation) or None
    """
    with get_cursor() as cur:
        cur.execute(
            "select patient_id, title, user_treatment_response, diagnosis_evaluation from conversations where id = %s",
            (conversation_id,),
        )
        row = cur.fetchone()
        if row:
            return (row[0], row[1] or "", row[2], row[3])
    return None


def update_conversation_treatment_response(conversation_id: str, user_response: str, evaluation: Optional[str] = None) -> None:
    """Update conversation with user's treatment response and evaluation."""
    with get_cursor() as cur:
        if evaluation:
            cur.execute(
                "update conversations set user_treatment_response = %s, diagnosis_evaluation = %s where id = %s",
                (user_response, evaluation, conversation_id),
            )
        else:
            cur.execute(
                "update conversations set user_treatment_response = %s where id = %s",
                (user_response, conversation_id),
            )


def get_patient_by_id(patient_id: str) -> Optional[Tuple[str, str, Optional[str], Optional[str]]]:
    """Get patient details by ID.
    
    Returns: (scenario, summary, treatment_plan, created_at_iso) or None
    """
    with get_cursor() as cur:
        cur.execute(
            "select scenario, summary, treatment_plan, to_char(created_at, 'YYYY-MM-DD" "HH24:MI:SSOF') from patients where id = %s",
            (patient_id,),
        )
        row = cur.fetchone()
        if row:
            return (row[0], row[1] or "", row[2], row[3])
    return None


def list_patient_conversations(patient_id: str) -> List[Tuple[str, str]]:
    with get_cursor() as cur:
        cur.execute(
            "select id, coalesce(title, '') from conversations where patient_id=%s order by created_at desc",
            (patient_id,),
        )
        rows = cur.fetchall()
    return [(r[0], r[1]) for r in rows]


def list_conversations_with_patient(limit: int = 50) -> List[Tuple[str, str, str, str, Optional[str]]]:
    """Return recent conversations joined with patient summary.

    Returns list of tuples: (conversation_id, created_at_iso, title, patient_summary, diagnosis_evaluation)
    """
    with get_cursor() as cur:
        cur.execute(
            """
            select distinct on (c.id) c.id,
                   to_char(c.created_at, 'YYYY-MM-DD" "HH24:MI:SSOF') as created_at,
                   coalesce(c.title, ''),
                   coalesce(p.summary, 'Brak podsumowania'),
                   c.diagnosis_evaluation
            from conversations c
            join patients p on p.id = c.patient_id
            order by c.id, c.created_at desc
            limit %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
    # Re-sort by created_at desc after distinct
    rows_sorted = sorted(rows, key=lambda x: x[1], reverse=True)[:limit]
    return [(r[0], r[1], r[2], r[3], r[4]) for r in rows_sorted]


def count_unprocessed_patients() -> int:
    """Count patients that don't have any completed conversation (with evaluation)."""
    with get_cursor() as cur:
        cur.execute(
            """
            select count(distinct p.id)
            from patients p
            where not exists (
                select 1
                from conversations c
                where c.patient_id = p.id
                and c.diagnosis_evaluation is not null
            )
            """,
        )
        row = cur.fetchone()
        return row[0] if row else 0


def get_unprocessed_patient_id() -> Optional[str]:
    """Get ID of a patient that doesn't have any completed conversation (with evaluation).
    
    Returns the oldest unprocessed patient ID, or None if all are processed.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            select p.id
            from patients p
            where not exists (
                select 1
                from conversations c
                where c.patient_id = p.id
                and c.diagnosis_evaluation is not null
            )
            order by p.created_at asc
            limit 1
            """,
        )
        row = cur.fetchone()
        return row[0] if row else None


def is_patient_processed(patient_id: str) -> bool:
    """Check if patient has at least one completed conversation (with evaluation)."""
    with get_cursor() as cur:
        cur.execute(
            """
            select count(*) > 0
            from conversations
            where patient_id = %s
            and diagnosis_evaluation is not null
            """,
            (patient_id,),
        )
        row = cur.fetchone()
        return row[0] if row else False


def wipe_all_data() -> None:
    """Dangerous: truncate all application tables (cascades, restart identity)."""
    with get_cursor() as cur:
        cur.execute(
            "truncate table messages, conversations, patients restart identity cascade"
        )


def save_interview_transcript(conversation_id: str, chunk_number: int, transcript_json: Dict) -> None:
    """Save interview transcript with LLM-assigned roles."""
    transcript_id = str(uuid.uuid4())
    import json
    with get_cursor() as cur:
        cur.execute(
            "insert into interview_transcripts (id, conversation_id, chunk_number, transcript_json) values (%s, %s, %s, %s)",
            (transcript_id, conversation_id, chunk_number, json.dumps(transcript_json)),
        )


def get_interview_transcripts(conversation_id: str) -> List[Tuple[int, Dict]]:
    """Get all transcripts for a conversation, ordered by chunk_number."""
    import json
    with get_cursor() as cur:
        cur.execute(
            "select chunk_number, transcript_json from interview_transcripts where conversation_id=%s order by chunk_number asc",
            (conversation_id,),
        )
        rows = cur.fetchall()
    result = []
    for row in rows:
        chunk_num = row[0]
        transcript_data = row[1]
        # PostgreSQL jsonb is already a dict, but handle both cases
        if isinstance(transcript_data, dict):
            result.append((chunk_num, transcript_data))
        elif isinstance(transcript_data, str):
            result.append((chunk_num, json.loads(transcript_data)))
        else:
            # Fallback: try to convert to dict
            result.append((chunk_num, dict(transcript_data) if transcript_data else {}))
    return result


def save_interview_suggestion(conversation_id: str, chunk_number: int, minute_number: int, suggestions: str) -> None:
    """Save LLM suggestions for a specific minute of recording."""
    suggestion_id = str(uuid.uuid4())
    with get_cursor() as cur:
        cur.execute(
            "insert into interview_suggestions (id, conversation_id, chunk_number, minute_number, suggestions) values (%s, %s, %s, %s, %s)",
            (suggestion_id, conversation_id, chunk_number, minute_number, suggestions),
        )


def get_interview_suggestions(conversation_id: str) -> List[Tuple[int, int, str]]:
    """Get all suggestions for a conversation, ordered by chunk and minute."""
    with get_cursor() as cur:
        cur.execute(
            "select chunk_number, minute_number, suggestions from interview_suggestions where conversation_id=%s order by chunk_number asc, minute_number asc",
            (conversation_id,),
        )
        rows = cur.fetchall()
    return rows


def update_conversation_interview_summary(conversation_id: str, summary: str, extracted_info: Optional[Dict] = None) -> None:
    """Update conversation with interview summary and extracted info."""
    import json
    with get_cursor() as cur:
        if extracted_info:
            cur.execute(
                "update conversations set interview_summary = %s, extracted_info = %s where id = %s",
                (summary, json.dumps(extracted_info), conversation_id),
            )
        else:
            cur.execute(
                "update conversations set interview_summary = %s where id = %s",
                (summary, conversation_id),
            )


def get_conversation_interview_data(conversation_id: str) -> Optional[Tuple[Optional[str], Optional[Dict]]]:
    """Get interview summary and extracted info for a conversation."""
    import json
    with get_cursor() as cur:
        cur.execute(
            "select interview_summary, extracted_info from conversations where id = %s",
            (conversation_id,),
        )
        row = cur.fetchone()
    if row:
        summary = row[0]
        extracted_info = json.loads(row[1]) if row[1] else None
        return (summary, extracted_info)
    return None


def mark_patient_skipped(patient_id: str) -> None:
    """Mark a patient as processed by creating a conversation with an evaluation.

    This is used to 'skip' a patient from the pool without conducting an interview.
    """
    conversation_id = str(uuid.uuid4())
    with get_cursor() as cur:
        # Create conversation with evaluation prefilled as SKIPPED
        cur.execute(
            "insert into conversations (id, patient_id, title, diagnosis_evaluation) values (%s, %s, %s, %s)",
            (conversation_id, patient_id, "Skipped", "SKIPPED (pominięty)"),
        )


