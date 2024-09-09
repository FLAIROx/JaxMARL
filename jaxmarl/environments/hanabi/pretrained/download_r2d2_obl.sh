# ensure you have installed git lfs before running this script: https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
git clone https://huggingface.co/mttga/obl-r2d2-flax "$SCRIPT_DIR/obl-r2d2-flax"
