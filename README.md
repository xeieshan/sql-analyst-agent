# sql-analyst-agent

An LLM agent that answers business questions about a SQLite database by inspecting the schema, testing queries and submitting a final SQL statement. It comes with an evaluation harness that grades the submitted SQL on its **result**, not its text, and logs every step of every run.

It runs on any model LiteLLM supports (OpenAI, Mistral, Gemini, Groq, local Ollama models, ...) and uses the [Chinook](https://github.com/lerocha/chinook-database) music-store database: 11 tables of artists, albums, tracks, customers, invoices and employees.

## Results

| Model | Exec. accuracy | Easy | Medium | Hard | LLM calls / q | Tool errors / q | Cost (USD) | p50 latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle (harness check) | 1.000 | 1.00 | 1.00 | 1.00 | 2.0 | 0.00 | 0.0000 | 0.00 |
| *your model here* | `make eval MODEL=...` | | | | | | | |

The **oracle** is a scripted fake model that calls one tool, then submits the gold SQL. It must score 100%, and CI checks that it does, so a failing model row means the model failed, not the grader. Real model rows are added by running `make eval MODEL=<litellm model>` followed by `make report`. No model numbers are listed until they come from a real run.

## How the agent works

```mermaid
sequenceDiagram
    participant U as Question
    participant M as LLM (via LiteLLM)
    participant T as ToolBox
    participant D as SQLite (read-only)
    U->>M: system prompt + question + tool specs
    loop up to 12 turns
        M->>T: tool call(s): list_tables / describe_table / run_sql
        T->>D: validated SELECT
        D-->>T: rows (max 20 shown) or error text
        T-->>M: tool result
    end
    M->>T: submit_answer(sql, answer)
    T->>D: final SQL must run before it is accepted
