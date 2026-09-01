# WiFi CSI Fall Detection Research Knowledge Base

Status: open
Label: wayfinder:map

## Destination

Create a human-readable, flat Markdown knowledge base under `research/` that synthesizes WiFi CSI and WiFi RSSI fall-detection research, links papers to topic summaries, records datasets and repositories, tracks open questions, and provides enough synthesized evidence to support later POC decisions about hardware, dataset, and evaluation protocol.

## Notes

- Domain: WiFi CSI and RSSI sensing for fall detection.
- Human readability is primary; stable headings, explicit filenames, indexes, and links must also support AI navigation.
- All knowledge-base documents stay directly under `research/`; no subdirectories.
- File prefixes: `paper-`, `topic-`, `dataset-`, and `repository-`.
- One paper gets one paper note; one topic gets one synthesis note.
- Use consistent comparison tables across topic notes.
- Prefer peer-reviewed papers, original datasets, and GitHub repositories.
- Abstract-only sources must be marked as limited analysis.
- Adjacent sensing methods are out of scope except WiFi RSSI.

## Decisions so far

- [Survey Existing WiFi CSI Fall-Detection Evidence](wifi-csi-fall-detection-01-evidence-landscape.md): Initial evidence landscape recorded; headline metrics remain provisional until full-paper verification.
- [Inventory CSI Fall-Detection Datasets and Repositories](wifi-csi-fall-detection-02-datasets-and-repositories.md): Five candidate resources recorded; direct POC suitability and access remain to be verified.
- [Research WiFall Fall-Detection Study](wifi-csi-fall-detection-06-wifall.md): WiFall added as an early CSI baseline; reported metrics and reproducibility details remain provisional.
- [Define Initial Research Topic Taxonomy](wifi-csi-fall-detection-03-topic-taxonomy.md): Seven flat topic notes established, with many-to-many links between topics and source notes.
- [Define POC Readiness Criteria](wifi-csi-fall-detection-04-poc-readiness-criteria.md): Research readiness is a human judgment with no fixed metric; synthesis uses all relevant available information.

## Not yet specified

- The point at which the project owner declares research synthesis ready for later POC decisions.
- Candidate hardware, dataset, and evaluation protocol.
- Later creation of `research/poc-decisions.md`.
- Final completeness criteria for `research/README.md` and `landscape.md`.

## Out of scope

- Radar, camera, mmWave, inertial, and other non-WiFi sensing modalities.
- Fall-risk prediction.
- Post-fall monitoring and alerting.
- Production deployment design.
