#!/bin/bash

PYTHON_FILE="convertFoodIngrMap.py"
ENV_NAME="temp_env_$(date +%s)" # Unique environment name
REQUIREMENTS_FILE="requirements_oldPandas.txt"

echo "Creating temporary conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.7 -y

# Run the installation and script within an activated subshell
# This is more robust than `conda run` if the shell is not perfectly configured.
(
    echo "Activating conda environment: $ENV_NAME"
    # Make sure conda activate is available
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"

    echo "Upgrading pip..."
    python -m pip install --upgrade pip

    echo "Installing packages from $REQUIREMENTS_FILE"
    if [ -f "$REQUIREMENTS_FILE" ]; then
        python -m pip install -r "$REQUIREMENTS_FILE"
    else
        echo "Warning: $REQUIREMENTS_FILE not found."
    fi

    echo "Running Python file: $PYTHON_FILE"
    python "$PYTHON_FILE"
)

# The subshell has exited, so the environment is deactivated.
# Now, remove the environment.
echo "Removing temporary conda environment: $ENV_NAME"
conda env remove -n "$ENV_NAME" -y

echo "Script finished."
