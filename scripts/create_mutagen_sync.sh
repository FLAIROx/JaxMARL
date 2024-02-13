#!/usr/bin/env bash

set -e

cmd=$1
if [ "$cmd" == "-h" ] || [ "$cmd" == "--help" ] ; then
    cat << EOF
Usage: `basename $0`
Description: (Re)create the mutagen link between your Mac and the devbox. 2-way sync between your Mac and the devbox. Runs on Mac only.
Options:
    -h/--help: Print usage.
EOF
    exit 0
fi

cwd=$(python -c "import os; print(os.path.dirname(os.path.realpath('${BASH_SOURCE[0]}')))")
root=$(dirname $cwd)

ignored_files=$(grep -o '^[^#]*' $root/.gitignore | awk '{ printf("%s,", $0) }')
ignored_files=${ignored_files%,}

mutagen_link_id="arrakis-jaxmarl"
ssh_devbox_name="arrakis"
if [ "$(mutagen sync list | grep $mutagen_link_id)" != "" ]; then
    mutagen sync terminate $mutagen_link_id
fi

mutagen sync create "$root" "ravi@$ssh_devbox_name:~/Documents/research/jaxmarl" \
    --name $mutagen_link_id \
    --sync-mode=two-way-safe \
    --ignore-vcs \
    --max-staging-file-size="10 MB" \
    --ignore="$ignored_files" \
    --default-file-mode=664 \
    --default-directory-mode=775
