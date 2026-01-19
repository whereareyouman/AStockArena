# Documentation Images

This folder contains images and diagrams used in the project documentation.

## Required Images

The following images are referenced in the documentation but need to be created:

### 1. System Overview (`system_overview.png`)
**Location**: Referenced in `README.md`
**Description**: High-level visualization of the AStock Arena system showing:
- Frontend dashboard interface
- Backend API layer
- Multiple trading agents
- Data flow between components
- External services integration

**Suggested Tool**: Draw.io, Figma, or similar
**Dimensions**: 1200x600px minimum

### 2. Performance Comparison (`performance_comparison.png`)
**Location**: Referenced in `README.md` (Experiments & Results section)
**Description**: Chart comparing performance metrics across different LLM models:
- Cumulative returns over time
- Sharpe ratios
- Win rates
- Maximum drawdowns

**Suggested Tool**: Python (matplotlib/seaborn), Excel, or data visualization tools
**Dimensions**: 1000x600px minimum

## Creating Images

### For Architecture Diagrams:
1. Use **Draw.io** (https://app.diagrams.net/)
2. Export as PNG with transparent background
3. Use consistent colors matching the project theme
4. Include legends where necessary

### For Performance Charts:
```python
# Example code to generate performance comparison
import matplotlib.pyplot as plt
import pandas as pd

# Your performance data_pipeline
data = {
    'Model': ['GPT-5.1', 'Claude 4.5', 'Gemini 2.5', 'DeepSeek', 'Qwen3'],
    'Returns': [0.12, 0.15, 0.10, 0.08, 0.11],
    'Sharpe': [1.2, 1.5, 1.0, 0.9, 1.1]
}

df = pd.DataFrame(data)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].bar(df['Model'], df['Returns'])
ax[0].set_title('Cumulative Returns')
ax[1].bar(df['Model'], df['Sharpe'])
ax[1].set_title('Sharpe Ratio')

plt.tight_layout()
plt.savefig('documentation/images/performance_comparison.png', dpi=300, bbox_inches='tight')
```

### For Screenshots:
1. Use high-resolution display settings
2. Capture clean, focused areas of the interface
3. Annotate key features if needed
4. Crop to relevant content

## Image Guidelines

- **Format**: PNG for diagrams, JPG for photos
- **Resolution**: Minimum 72 DPI for web, 300 DPI for print
- **Size**: Optimize for web (< 500KB each)
- **Naming**: Use lowercase with hyphens (e.g., `system-architecture.png`)
- **Alt Text**: Ensure README references include descriptive alt text

## Current Placeholders

The following images are currently placeholders and should be created:

- [ ] `system_overview.png` - Main system architecture visualization
- [ ] `performance_comparison.png` - Model performance comparison chart
- [ ] `dashboard_screenshot.png` - Frontend dashboard screenshot (optional)
- [ ] `agent_decision_flow.png` - Agent decision-making flowchart (optional)

## Tools & Resources

**Diagram Tools:**
- Draw.io / diagrams.net
- Lucidchart
- Figma
- Microsoft Visio

**Chart Tools:**
- Python (matplotlib, seaborn, plotly)
- R (ggplot2)
- Excel / Google Sheets
- Tableau

**Screenshot Tools:**
- macOS: Cmd+Shift+4
- Windows: Win+Shift+S
- Linux: gnome-screenshot, flameshot

## Contributing Images

When adding new images:
1. Place them in this folder
2. Use descriptive filenames
3. Update this README with description
4. Reference them in documentation with relative paths
5. Ensure images are compressed/optimized

---

**Note**: Images are not required for the system to function, but they significantly improve documentation quality for academic publications and open-source presentation.
