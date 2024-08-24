# Contributing to MM Toolbox
First off, thank you for considering contributing to MM Toolbox! Your help is essential to maintaining and improving this project. I appreciate your interest and efforts.

## How to Contribute

### 1. Reporting Issues
If you encounter any bugs, issues, or have feature requests, please open an issue on GitHub. Provide as much detail as possible, including:

- A clear and descriptive title
- A detailed description of the problem or feature request
- Steps to reproduce the issue (if applicable)
- Code snippets or screenshots (if helpful)
- The environment you’re using (e.g., Python version, OS, etc.)

### 2. Fork and Clone the Repository
To start contributing:

1. Fork the repository to your GitHub account.
2. Clone your forked repository to your local machine:
   ```bash
   git clone https://github.com/beatzxbt/mm-toolbox.git
   cd mm-toolbox
   ```

### 3. Create a Branch
Create a new branch for your contribution:
```bash
git checkout -b feature-or-bugfix-name
```

Make sure to choose a descriptive name for your branch that reflects the changes you intend to make.

### 4. Install Dependencies
Install these by doing:
```bash
pip install -r requirements.txt
```

### 5. Make Your Changes
Implement your changes in the appropriate module(s). Be sure to format your code with black and add numpy style docstrings where neccesary. If possible, avoid introducing new dependencies.

### 6. Write Tests
Ensure that your changes are well-tested. If you add new functionality, write corresponding unit tests. Tests are located in the tests/ directory.

Run the tests to ensure everything is working correctly:
```bash
python -m unittest discover tests
```

### 7. Commit Your Changes
Before committing, please make sure your code follows the project's coding standards. Use descriptive commit messages:
```bash
git add .
git commit -m "Descriptive message about your changes"
```

### 8. Push to Your Fork
Push your changes to your forked repository:
```bash
git push origin feature-or-bugfix-name
```

### 9. Create a Pull Request
Go to the original repository on GitHub and create a Pull Request (PR) from your forked repository.

In your PR:
* Provide a clear and descriptive title.
* Explain the purpose and details of your changes.
* Mention any related issues or PRs.
* Highlight any particular areas you'd like reviewers to focus on.

### 10. Address Feedback
Your PR will be reviewed by the maintainers. You may be asked to make additional changes based on feedback. Please address these promptly to move the process forward.

## Code Style
* Follow PEP 8 for Python code style ([black](https://github.com/psf/black) makes this very simple).
* Keep functions and classes well-documented with clear and concise comments.
* Use descriptive variable and function names.
* Write unit tests for all new features and ensure existing tests pass.

## License
By contributing to MM Toolbox, you agree that your contributions will be licensed under the MIT License.
