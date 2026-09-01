# Domain Docs

Use single-context domain documentation:

```text
/CONTEXT.md
/docs/adr/
```

Before exploring code:

1. Read root `CONTEXT.md` if present.
2. Read ADRs under `docs/adr/` relevant to the work.
3. Proceed silently when either location does not exist.
4. Use glossary terms from `CONTEXT.md`.
5. Surface contradictions with existing ADRs.
6. Create `CONTEXT.md` and ADRs lazily when domain terms or durable decisions are resolved.
