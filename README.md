# Academic Research Skills — Personal Working Copy

This repository is a personal working copy of the **Academic Research Skills for OpenCode** project. It is retained as tooling used to explore structured research workflows, literature review, manuscript checking, and reproducibility-oriented academic assistance.

## Important Attribution

The underlying project is not my original framework. The repository content identifies the OpenCode port by `timpara/opencode-academic-research`, based on `timpara/academic-research-skills`, with the original Claude Code plugin by Cheng-I Wu (`Imbad0202/academic-research-skills`).

The upstream projects, authors, license, DOI, and contributor history should be cited and preserved when using or redistributing this material.

## How I Use It

I keep this repository as a working environment for experimenting with structured research-assistance workflows around:

- literature review and evidence organization;
- manuscript consistency checks;
- citation verification;
- experiment provenance and reproducibility notes;
- reviewer-response workflows.

## Academic Portfolio Note

This repository is included as research tooling rather than as a claim of authorship of the underlying agent/skill framework. My original research and engineering projects are maintained in separate repositories on this profile.


## Goal

This working copy is used to study repeatable literature-review, manuscript-audit, citation, and reviewer-response procedures around an upstream academic-research toolset.

## Installation

The repository uses Python tooling for validation and Node.js for type checking. With `uv` and Node.js installed:

```bash
uv sync --extra dev
npm install
npm run typecheck
```

Consult the upstream setup documentation for installing the OpenCode skills and commands themselves; directory placement depends on the OpenCode configuration in use.

## Working with the Repository

Skill definitions, commands, and supporting agents follow the upstream layout. Run the repository's validation scripts and tests before modifying a workflow, and keep `NOTICE.md`, `CITATION.cff`, `LICENSE`, and contributor history intact. Local use or experimentation does not transfer authorship of the framework.
