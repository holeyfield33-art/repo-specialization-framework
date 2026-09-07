"""Repository Specialization Experimental Framework (RSEF).

Research harness for testing whether small coding models become substantially
more effective on one repository when combined with:
  - repo-specific QLoRA/SFT
  - SHA-bound structured file packs
  - deterministic dependency graph
  - held-out historical change events
  - automated verification

Target question (falsifiable):
Does repository-specific adaptation + structured file knowledge + graph context
outperform the same untuned small model using ordinary repository-wide context?
"""

__version__ = "0.1.0"
__target_repo__ = "holeyfield33-art/runtime-firewall-mvp"
