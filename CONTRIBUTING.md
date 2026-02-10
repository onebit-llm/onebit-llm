# Contributing to OneBit-LLM

Thank you for your interest in contributing. We welcome contributions: code, documentation, issues, and ideas.

## How to contribute

### Report bugs or suggest features

- **Bugs:** Open an issue with the **Bug report** template. Include steps to reproduce, your environment (OS, Python version, PyTorch/CUDA if used), and relevant config.
- **Features / ideas:** Open an issue with the **Feature request** template. Discussion before coding is encouraged so we can align on design.

### Contribute code or docs

1. **Fork** the repository on GitHub.
2. **Clone** your fork and create a branch from `main`:
   ```bash
   git clone https://github.com/YOUR_USERNAME/onebit-llm.git
   cd onebit-llm
   git checkout -b my-feature
   ```
3. **Set up the project:**
   ```bash
   pip install -e .
   ```
4. **Make your changes.** Keep commits focused and messages clear.
5. **Run checks:** Install and run tests if present (`pytest`), and ensure the stack-stream script runs with a small config if you changed training or model code.
6. **Push** to your fork and open a **Pull Request** against `main`. Use the PR template and reference any related issues.

We’ll review as soon as we can. You may be asked for small edits; once approved, a maintainer will merge.

### Join the organization

If you’d like to become a member of the **onebit-llm** organization:

- Open an issue with the title **"Request to join onebit-llm org"** and briefly say how you’d like to contribute, or
- Contact the maintainers (e.g. via an issue or the contact listed in the org profile).

Thank you for contributing.
