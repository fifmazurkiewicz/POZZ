# Interview controls

Status: approved in conversation; voice ID setting added by explicit user request.

Both simulated and manually authored interviews provide Stop, Examination and End interview controls in Polish. Stop immediately silences audio, aborts client requests, discards unfinished microphone input and invalidates late responses. Cancellation of a browser request cannot guarantee cancellation of a synchronous provider job already running on the server.

Examination opens an accessible dialog asking which examination or test to perform. The server generates a scenario-consistent result, stores it in conversation history and does not reveal hidden diagnosis or treatment gold. Results are text, not patient speech.

End interview stops audio and opens a treatment-plan dialog. Submitting generates and persists an evaluation using hidden reference gold, then marks the conversation ended. Failed evaluation leaves the conversation open for retry. Completed conversations reject new turns and examinations and remain readable. Where no gold exists, generate a reference from the scenario without consulting the doctor's plan. No new database columns are required.

Menu includes an ElevenLabs voice ID preference saved on this browser. Empty means server default. Validate the ID on both client and server; never accept arbitrary URLs. Browser speech fallback cannot use an ElevenLabs ID. Preserve existing clinical styling and use native dialogs with keyboard dismissal, focus management, visible labels and inline errors.

The checked-out implementation uses chained TTS, not an active Gemini Live socket. Implement stop against the actual playback path, without claiming to implement a Live integration.
