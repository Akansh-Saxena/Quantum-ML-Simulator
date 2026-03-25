#!/bin/bash
git filter-branch -f --env-filter '
MSG=$(git log -1 --pretty=format:%B $GIT_COMMIT)
if echo "$MSG" | grep -iqE "DONE|LEFT"; then
    # Do nothing, keep original author
    true
else
    export GIT_AUTHOR_NAME="Akansh Saxena"
    export GIT_AUTHOR_EMAIL="saxenaakansh29@gmail.com"
    export GIT_COMMITTER_NAME="Akansh Saxena"
    export GIT_COMMITTER_EMAIL="saxenaakansh29@gmail.com"
fi
' -- --all
