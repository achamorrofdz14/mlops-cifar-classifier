"""Module to get the Git username from the global configuration."""

import subprocess


def get_git_user_name() -> str:
    """Get the Git username from the global configuration."""
    try:
        # Get the Git username from the global configuration
        return (
            subprocess.check_output(["git", "config", "user.name"])
            .strip()
            .decode("utf-8")
        )
    except subprocess.CalledProcessError:
        # Return a default value if Git user is not found
        return "unknown_user"
