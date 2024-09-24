# MLOps CIFAR Classification

## Description
This repository demonstrates MLOps practices with a CIFAR-10 image classification model, covering data preprocessing, model training, deployment, and monitoring. Ideal for understanding and applying MLOps in real-world scenarios.


## Branch Types and Their Purposes

### Feature / Fix / Experiment Branches

* **Purpose:** Used for active development work, including:
    * Implementing new features
    * Fixing bugs
    * Conducting experiments with different solutions
* **Team Members:** Data scientists, ML engineers, and software developers collaborate on these branches.
* **Naming Conventions:**
    * `feature/<feature-name>` (e.g., `feature/data-processing`)
    * `fix/<issue-description>` (e.g., `fix/corrupted-package`)
    * `experiment/<experiment-name>`
* **Actions on Push:**
    * Pre-commit hooks with linters (Ruff, Black) are automatically executed to ensure code quality.
    * If the commit message is "run tests," unit tests (pytest) are also triggered in addition to pre-commit checks.

### Develop Branch

* **Purpose:** Serves as an integration point for all feature, fix, and experiment branches before they are promoted to the release stage.
* **Key Points:**
    * This branch is protected to maintain code stability.
    * Merge requests into develop trigger the following:
        * Integration tests are executed to ensure that the combined code from different branches works seamlessly together.
        * A code review process is in place for approval.
        * Docker builds are created for consistent deployment environments.

### Release Branch

* **Purpose:** When the develop branch is considered ready for release, it is merged into the release branch.
* **Key Points:**
    * Merging into release unlocks the develop branch, allowing development to continue on new features while the current version is prepared for deployment.
    * This branch is typically used for deployment to a staging environment, where a replica of the production setup is used for testing.
    * In this project, a local architecture might be used for staging due to specific project constraints.

### Main (Production) Branch

* **Purpose:** This is the live production branch, representing the code that is actively running in the production environment.
* **Key Points:**
    * Merge requests into main trigger a GitLab CI/CD pipeline, automating the process of replacing the existing production code with the new version from the release branch.
    * High levels of protection and scrutiny are expected for merges into this branch.

## Additional Considerations

### Hotfix Branches

* **Purpose:** Used to address critical issues in the production environment that require immediate attention.
* **Naming Convention:** `hotfix/<issue-description>`
* **Setps:**
    * Branch off from the `main` branch.
    * Apply the fix and thoroughly test it.
    * Merge the hotfix branch back into both `main` and `develop` to ensure the fix is included in future releases.
    * Create a new release from `main` after applying the hotfix.

### Tooling

* **GitLab:** Used for version control and CI/CD pipelines.
* **Ruff & Black:** Python linters for code style enforcement.
* **pytest:** Framework for unit testing.
