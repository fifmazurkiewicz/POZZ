# POZZ — business flow: recorded interview analysis

```mermaid
flowchart TD
    start([Doctor starts a recorded interview]) --> record[Record or upload interview audio]
    record --> consent{Consent and permitted use confirmed?}
    consent -- No --> stop([Do not process audio])
    consent -- Yes --> upload[Send audio for secure processing]
    upload --> stt[Speech-to-text transcription]
    stt --> extract[LLM extracts clinical facts and draft summary]
    extract --> review[Doctor reviews and edits the draft]
    review --> save[Save approved transcript, summary and recommendations]
    save --> discard[Discard source audio under retention rules]
    discard --> end([Interview record available])
```

The doctor remains responsible for reviewing any generated clinical content.
