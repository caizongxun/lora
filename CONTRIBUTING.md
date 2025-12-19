# Contributing to LoRA Fine-Tuning Evaluation

Thank you for your interest in contributing! Please follow these guidelines to help us maintain code quality and consistency.

## Code of Conduct

Be respectful and constructive in all interactions with other contributors and maintainers.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/lora.git
   cd lora
   ```
3. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8
   ```

## Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes and test them locally
3. Follow PEP 8 coding standards:
   ```bash
   black *.py
   flake8 *.py
   ```
4. Add documentation for new features
5. Update the README.md if needed

## Commit Messages

Use clear, descriptive commit messages:

- Good: "Add radar chart visualization for performance comparison"
- Bad: "fix stuff"

Format: `Type: Description`

Types:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `refactor:` - Code refactoring without behavior changes
- `test:` - Test additions or modifications
- `perf:` - Performance improvements

## Testing

If adding new functionality:
1. Add appropriate error handling
2. Test with different dataset sizes
3. Verify GPU/CPU compatibility
4. Test with different model configurations

## Pull Request Process

1. Ensure your code follows the project style
2. Update documentation and comments
3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
4. Submit a Pull Request with:
   - Clear title describing the change
   - Description of what was changed and why
   - Reference to any related issues
   - Mention any breaking changes

## Code Style Guidelines

### Python
- Follow PEP 8 standards
- Use type hints for function arguments and returns
- Add docstrings to all functions and classes
- Use descriptive variable names
- Maximum line length: 100 characters

Example:
```python
def evaluate_model(
    model_name: str,
    datasets_dict: Dict[str, Dict[str, List]]
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate model performance on datasets.
    
    Args:
        model_name (str): Model identifier
        datasets_dict (dict): Dictionary containing datasets
        
    Returns:
        results (dict): Evaluation results
    """
    # Implementation
    pass
```

## Documentation

- Add docstrings to all new functions
- Include type hints
- Update README.md for new features
- Add comments for complex logic
- Use breakpoint markers for key sections (e.g., [BREAKPOINT_1])

## Reporting Bugs

When reporting bugs, include:
1. Python version
2. PyTorch and CUDA version
3. Error message and traceback
4. Steps to reproduce
5. Expected behavior vs actual behavior

## Feature Requests

Describe:
1. What problem the feature solves
2. How you envision the solution
3. Possible alternatives or related features

## Questions?

Feel free to open an issue with the label `question` or start a discussion.

## Review Process

Contributions will be reviewed within 1-2 weeks. Feedback will be provided for improvements if needed.

Thank you for contributing!
