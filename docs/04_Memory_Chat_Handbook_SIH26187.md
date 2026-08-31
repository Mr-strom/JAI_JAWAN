# Continuity Protocol & Chat Handoff Handbook — SIH26187

**Version:** 1.0 | 2026-08-31 **Classification:** SIH Internal Round — Top 50 Qualifier **Status:** **ACTIVE PROTOCOL**

> **Document Links:** [[01_PRD]] · [[02_TRD]] · [[03_Architecture_Map]] · [[05_Project_Handbook]] · [[06_Version_Control_Testing_Strategy]] **Purpose:** Ensure zero context loss when switching between AI chat sessions. Every chat session has tool budget/token limits; when exhausted, this handbook guarantees the next session picks up exactly where the previous one left off — no rework, no re-explaining, no lost decisions. This is a standing instruction that should be re-stated at the start of every new session.

## Section 1: The Handoff Trigger

When ANY of the following occur, the assistant must stop normal work and output a compressed Child Document/Handoff Package — do not wait for the user to ask.

**Trigger Conditions:**

- **Token / Tool Budget Exhausted:** Current session's remaining token capacity/credits drops to ~10%, or tool calls hit the ~20-25 limit.
    
- **Context Decay:** The context window is approaching its limit (model starts forgetting early messages).
    
- **User Command:** User explicitly says "new chat" or "continue in fresh chat".
    
- **File Limits:** A file write operation fails due to size limits (must split large files), or a complex multi-file generation task is incomplete at session end.
    

> **Note for the Human Operator (Proxy Triggers):** No chat platform exposes a live "% tokens remaining" readout to the assistant mid-session. In practice, treat these as trigger proxies: (a) You notice the assistant's responses getting shorter or it starts summarizing unprompted. (b) The conversation has produced ~15–20 substantial exchanges since the last checkpoint. (c) You're about to start a new major phase (see [[06_Version_Control_Testing_Strategy]] phase cities). _When in doubt, ask for the Child Document proactively._

## Section 2: Chat Budget & File Generation Awareness

- **Typical session limit:** ~20-25 tool calls (read, write, search, shell, etc.).
    
- **File generation planning:**
    
    - _Small files (<50KB):_ Can write 3-4 per session.
        
    - _Large files (>100KB):_ Write 1-2 per session, or split into parts.
        
    - _Batch generation:_ If generating 8 files, expect to need 2-3 sessions minimum.
        
- **Monitoring:** After 15 tool calls, start planning the handoff.
    

## Section 3: The Handoff Package (Child Document Template)

When triggered, output exactly this structure, nothing more. Every handoff MUST include this package.

Markdown

```
# SIH26187 — CHILD CONTINUITY DOCUMENT
**Generated:** [Date/Time] | **Version:** [X.Y]
**Parent Session:** [Brief description, e.g., "Model A build session 2"]
**Tool Budget Status:** [Exhausted / Nearing limit]

## 1. Locked Baseline Reference
This project's immutable architecture lives in the Master Handoff + [[01_PRD]] through [[07B_Model_B_Delegation]].
*Master Handoff Location:* `/mnt/agents/temp/SIH26187_MASTER_HANDOFF.md`
**Do not redesign anything marked LOCKED without explicit owner approval.**

## 2. Current Build State & File Inventory
*   **What is built and working:** [List]
*   **What is built but broken/untested:** [List]
*   **What is not started:** [List]

| # | Filename | Status | Size | Notes |
|---|---|---|---|---|
| 1 | 01_PRD_SIH26187.md | COMPLETE | [bytes] | Full PRD written |
| 2 | 02_TRD_SIH26187.md | PARTIAL | [bytes] | Header only, content pending |
| 3 | [Filename] | PENDING | — | Not started |

*(Status Key: COMPLETE = full content written/verified | PARTIAL = incomplete content | PENDING = not started)*

## 3. Codebase Pointers
*   **Repo/Branch:** [Name — see 06_Version_Control_Testing_Strategy]
*   **Key files touched this session:** [Paths]
*   **Schema version in use:** [schema_v_N, per 02_TRD JSON Event Schema]

## 4. Unresolved Issues & Open Questions
1. [Issue/Question] — Blocking: [Yes/No]
2. [Copy forward any unresolved items from handoff/Project Handbook awaiting Owner's Call]

## 5. Decisions Made This Session
1. [Decision] — Rationale: [Why]
*(Only list new decisions made in this session that are not yet in the master handoff)*

## 6. Immediate Next Step
[The single next action the new session should take, stated as one sentence. E.g., "Continue file generation from where this session left off."]
```

## Section 4: Continuity Rules & Constraints

### 4.1 Rules for the Child Document Output

1. **Compression, not omission.** Every unresolved issue and every LOCKED decision must survive the compression. What gets cut is exploratory conversation, not facts.
    
2. **Never silently drop a blocking issue.** If something was blocking and remains unresolved, it must appear in Section 4 of every Child Document until it's resolved.
    
3. **Never re-litigate a LOCKED decision** inside a Child Document. If the new session's AI proposes changing something LOCKED, the human operator should treat that as a hallucination risk, not a valid suggestion — cross-check against `03_Architecture_Map` first.
    

### 4.2 What NOT to Hand Off

Do NOT include the following in the handoff package:

- Internal AI reasoning or tool call logs.
    
- Failed experiments or dead-end approaches (unless they contain specific lessons).
    
- Duplicate copies of files already saved to disk.
    
- Temporary scratchpad content.
    
- Anything marked PROPOSED unless it became a confirmed decision in this session.
    

### 4.3 File Naming Convention (Locked)

All project files MUST follow this pattern: `XX_DescriptiveName_SIH26187.md`

- Where `XX` is a zero-padded two-digit number for ordering.
    
- **NEVER** use spaces in filenames. **ALWAYS** use the `SIH26187` suffix. **ALWAYS** use the `.md` extension.
    
- _Examples:_ `01_PRD_SIH26187.md`, `07a_ModelA_Dev_Guide_SIH26187.md`
    

### 4.4 Version Control for Handoffs

Each handoff session gets a version bump, tracked in the master handoff header and each file's version field:

- `v1.0` — Initial master handoff creation.
    
- `v1.1` — First file generation session.
    
- `v1.2` — Architecture diagrams added. (etc.)
    

## Section 5: The Carry-Over Message & Human Operator Checklist

### 5.1 The Carry-Over (Paste-and-Go)

The Child Document should be the **first thing pasted** into a new session, before any new instruction, so the new AI's context is seeded correctly. When starting a new session, the receiving AI MUST:

- Read the master handoff first.
    
- Read any COMPLETE or PARTIAL files to understand current state.
    
- Resume next actions without asking for information already provided.
    
- Never redesign, restructure, or improve architecture without explicit owner approval.
    

### 5.2 Session Handoff Checklist (For Human Operator)

Before closing the old session and starting a new one, verify:

- [ ] All decisions are written down (not just spoken in chat).
    
- [ ] No critical information exists only in chat memory — it must be in a file or the Child Document.
    
- [ ] Child Document generated and copied out of the old session.
    
- [ ] New session started.
    
- [ ] Child Document pasted as the **first message**.
    
- [ ] New AI asked to confirm understanding (yes/no, per established pattern) before continuing work.
    
- [ ] Old session archived (not deleted) in case a fact was missed.
    

### 5.3 Recovery from a Broken Handoff

If a new session starts WITHOUT the handoff package:

1. Check `/mnt/agents/temp/` for `SIH26187_MASTER_HANDOFF.md`.
    
2. Check `/mnt/agents/output/` for any existing files.
    
3. Reconstruct file inventory from disk.
    
4. **AI Must Ask User:** "I found [X] files from the previous session. Should I continue from there or restart?"
    
5. _Never assume a restart is desired — always confirm._
    

**Document Control** **Owner:** Project Owner. | **Reviewers:** All team members using AI assistance. | **Change Log:** v1.0 — 2026-08-31 — Initial protocol from session recovery experience.