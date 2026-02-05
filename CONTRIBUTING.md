# Contributing to OneBit-LLM

Thank you for your interest in contributing. We welcome contributions from everyone: code, documentation, issues, and ideas.

---

## How to contribute

### Report bugs or suggest features

- **Bugs:** Open an [issue](https://github.com/onebit-llm/onebit-llm/issues) with the **Bug report** template. Include steps to reproduce, your environment (OS, Rust version, CUDA if used), and relevant config.
- **Features / ideas:** Open an issue with the **Feature request** template. Discussion before coding is encouraged so we can align on design.

### Contribute code or docs

1. **Fork** the repository on GitHub.
2. **Clone** your fork and create a branch from `main`:
   ```bash
   git clone https://github.com/YOUR_USERNAME/onebit-llm.git
   cd onebit-llm
   git checkout -b my-feature
   ```
3. **Make your changes.** Keep commits focused and messages clear.
4. **Run tests and format:**
   ```bash
   cargo test --workspace
   cargo fmt --all
   cargo clippy --workspace
   ```
5. **Push** to your fork and open a **Pull Request** against `main`. Use the PR template and reference any related issues.

We’ll review as soon as we can. You may be asked for small edits; once approved, a maintainer will merge.

### Join the organization

If you’d like to become a member of the **onebit-llm** organization (e.g. to help triage issues or manage repos):

- Open an issue with the title **"Request to join onebit-llm org"** and briefly say how you’d like to contribute, or
- Contact the maintainers (e.g. via an issue or the contact listed in the org profile).

There is no obligation to join the org to contribute; fork + PR is enough and very welcome.

---

## Code and project guidelines

- **Rust:** Follow standard Rust style. Run `cargo fmt` and fix `cargo clippy` warnings when possible.
- **Scope:** OneBit-LLM is a 1-bit / ternary LLM framework in Rust using Candle. The workspace has five crates (`ternary-core`, `ternary-common`, `ternary-train`, `ternary-search`, `ternary-infer`). We prefer changes that fit this scope.
- **Docs:** New public APIs and non-obvious behavior should be documented. User-facing changes should be reflected in README or `docs/MASTER_PLAN.md` when relevant.
- **Tests:** Bug fixes are easier to merge when they add or extend tests. New features are welcome to include tests where practical. Run `cargo test --workspace` to verify.

---

## Where to start

- **Good first issues:** Look for issues labeled `good first issue` (if any).
- **Documentation:** Fixing typos, clarifying README or `docs/`, or adding examples.
- **Small improvements:** Config defaults, CLI help text, error messages, or dependency updates.

If you’re unsure, open an issue and ask; we’re happy to help.

---

## Code of conduct

We expect respectful and constructive behavior. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for the full text.

---

## Questions

- Open a [Discussion](https://github.com/onebit-llm/onebit-llm/discussions) (if enabled) or an **Issue** with the question label.
- For security-sensitive topics, prefer contacting the maintainers privately (e.g. via GitHub or the org’s contact).

Thank you for contributing.
