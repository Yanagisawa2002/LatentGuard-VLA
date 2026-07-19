# M6A blocker registry

Overall M6B readiness is **blocked**: 2 critical, 4 major, and 0 minor
blockers. Missing evidence is not downgraded.

| ID | Severity | Owner | Blocker | Required resolution |
| --- | --- | --- | --- | --- |
| M6A-B003 | critical | LatentGuard-VLA | Candidate source and blind pool are not frozen. | Choose one source, count, seed semantic, and pre-outcome manifest. |
| M6A-B004 | critical | LangMani | ACT queue/history has no inference snapshot. | Serialize complete policy state or constrain M6B to an initial episode boundary. |
| M6A-B005 | major | LatentGuard-VLA | Compact verifier input is not frozen. | Freeze a minimal non-privileged allowlist. |
| M6A-B006 | major | LangMani | Intermediate restored-state paired execution lacks accepted evidence. | Validate it or use seeded initial resets only. |
| M6A-B007 | major | LatentGuard-VLA | Python/NumPy pins are not in-process compatible. | Validate the canonical JSON process bridge. |
| M6A-B009 | major | LatentGuard-VLA | One task/checkpoint entry is not frozen. | Bind it before seeds or outcomes are available. |

The machine-readable source of truth is [`blockers.json`](blockers.json).
No workaround listed here is authorization to execute M6B.
