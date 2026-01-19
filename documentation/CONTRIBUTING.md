# Contributing to AStock Arena

Thank you for your interest in contributing to AStock Arena! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- **Be respectful**: Treat all community members with respect and courtesy
- **Be collaborative**: Work together to achieve the best outcomes
- **Be inclusive**: Welcome diverse perspectives and experiences
- **Be constructive**: Provide helpful feedback and accept it graciously

## Getting Started

### Prerequisites

Before contributing, ensure you have:

1. **Python 3.10+** installed
2. **Node.js 16.x+** for frontend development
3. **Git** for version control
4. A **GitHub account**

### Setting Up Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/your-username/AStockArena.git
cd AStockArena

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists

# Install pre-commit hooks (recommended)
pre-commit install

# Set up frontend
cd Tradingsimulation
npm install
cd ..
```

### Configuration

Copy the environment template and configure your credentials:

```bash
cp utilities/shell/env.sh.template utilities/shell/env.sh
# Edit env.sh with your API keys
source utilities/shell/env.sh
```

## Development Workflow

### 1. Create a Branch

Always create a new branch for your work:

```bash
# Feature branch
git checkout -b feature/your-feature-name

# Bug fix branch
git checkout -b fix/bug-description

# Documentation branch
git checkout -b docs/what-you-document
```

### 2. Make Changes

- Write clear, self-documenting code
- Add comments for complex logic
- Update documentation as needed
- Write or update tests for new functionality

### 3. Test Your Changes

```bash
# Run Python tests
pytest tests/

# Run linting
flake8 .
black --check .

# Test frontend
cd Tradingsimulation
npm test
npm run lint
cd ..
```

### 4. Commit Your Changes

Follow conventional commit message format:

```bash
git add .
git commit -m "feat: add new trading indicator"

# Commit types:
# feat: New feature
# fix: Bug fix
# docs: Documentation changes
# style: Code style changes (formatting, etc.)
# refactor: Code refactoring
# test: Adding or updating tests
# chore: Maintenance tasks
```

### 5. Push and Create Pull Request

```bash
git push origin your-branch-name
```

Then create a Pull Request on GitHub.

## Coding Standards

### Python

- **Style Guide**: Follow [PEP 8](https://peps.python.org/pep-0008/)
- **Formatter**: Use `black` for code formatting
- **Linter**: Use `flake8` for linting
- **Type Hints**: Use type hints for function signatures
- **Docstrings**: Use Google-style docstrings

Example:

```python
def calculate_sharpe_ratio(
    returns: List[float], 
    risk_free_rate: float = 0.0
) -> float:
    """Calculate the Sharpe ratio for a series of returns.
    
    Args:
        returns: List of period returns
        risk_free_rate: Risk-free rate of return
        
    Returns:
        The calculated Sharpe ratio
        
    Raises:
        ValueError: If returns list is empty
    """
    if not returns:
        raise ValueError("Returns list cannot be empty")
    
    excess_returns = [r - risk_free_rate for r in returns]
    return np.mean(excess_returns) / np.std(excess_returns)
```

### JavaScript/TypeScript

- **Style Guide**: Follow project ESLint configuration
- **Formatter**: Use Prettier
- **Types**: Use TypeScript for type safety

Example:

```typescript
interface Position {
  symbol: string;
  quantity: number;
  entryPrice: number;
  currentPrice: number;
}

function calculatePnL(position: Position): number {
  return (position.currentPrice - position.entryPrice) * position.quantity;
}
```

### File Organization

- **Imports**: Group imports (stdlib, third-party, local)
- **Constants**: Define at module level in UPPER_CASE
- **Functions**: One responsibility per function
- **Classes**: Clear separation of concerns

## Testing Guidelines

### Unit Tests

Write unit tests for all new functionality:

```python
# tests/test_indicators.py
import pytest
from tools.indicators import calculate_rsi

def test_rsi_calculation():
    """Test RSI calculation with known values."""
    prices = [44, 45, 46, 47, 48, 49, 50]
    rsi = calculate_rsi(prices, period=6)
    assert 0 <= rsi <= 100
    assert rsi > 50  # Prices are trending up
```

### Integration Tests

Test component interactions:

```python
# tests/test_agent_integration.py
def test_agent_decision_flow():
    """Test complete agentic workflow decision flow."""
    agent = create_test_agent()
    snapshot = fetch_test_snapshot()
    decision = agent.make_decision(snapshot)
    assert decision.action in ['BUY', 'SELL', 'HOLD']
    assert 0 <= decision.confidence <= 1.0
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_indicators.py

# Run with coverage
pytest --cov=. --cov-report=html

# Run frontend tests
cd Tradingsimulation
npm test
```

## Pull Request Process

### Before Submitting

1. **Update documentation** if you've changed APIs or functionality
2. **Add tests** for new features
3. **Run all tests** and ensure they pass
4. **Update CHANGELOG.md** with your changes
5. **Rebase** on latest main branch

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe how you tested these changes

## Checklist
- [ ] Code follows project style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests passing
- [ ] No new warnings
```

### Review Process

1. Maintainers will review your PR within 3-5 business days
2. Address any feedback or requested changes
3. Once approved, a maintainer will merge your PR

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
**Describe the bug**
Clear description of what the bug is

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen

**Screenshots**
If applicable, add screenshots

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10.5]
- Node version: [e.g., 16.14.0]

**Additional context**
Any other relevant information
```

### Feature Requests

Use the feature request template:

```markdown
**Is your feature request related to a problem?**
Description of the problem

**Describe the solution you'd like**
Clear description of what you want to happen

**Describe alternatives you've considered**
Other solutions you've thought about

**Additional context**
Any other relevant information
```

## Project Structure

Understanding the codebase organization:

```
AStockArena/
├── agent_engine/        # Trading agents
├── api_server.py        # FastAPI backend
├── settings/            # Configuration files (primary)
├── configs/             # Configuration files (alternative location)
├── data_flow/           # Data storage and pipeline
│   ├── data_pipeline.py # Data pipeline orchestration
│   ├── pnl_snapshots/   # PnL snapshots
│   └── debug/           # Debug logs
├── documentation/       # Documentation
├── prompt_templates/    # LLM prompts
├── utilities/           # Utility scripts
├── utils/               # Helper modules
├── analysis/            # Analysis and visualization
├── experiments/         # Alternative analysis workspace
└── Tradingsimulation/   # Frontend
```

## Getting Help

If you need help:

1. **Check documentation** in the `documentation/` folder
2. **Search existing issues** on GitHub
3. **Join discussions** in GitHub Discussions
4. **Ask questions** by creating a new issue with the "question" label

## Recognition

Contributors will be:
- Listed in the project README
- Mentioned in release notes
- Given credit in any related publications (if significant contribution)

---

Thank you for contributing to AStock Arena! Your efforts help advance research in LLM-based trading systems.
