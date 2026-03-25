#!/bin/bash
git filter-branch -f --env-filter '
    export GIT_AUTHOR_NAME="Akansh Saxena"
    export GIT_AUTHOR_EMAIL="saxenaakansh29@gmail.com"
    export GIT_COMMITTER_NAME="Akansh Saxena"
    export GIT_COMMITTER_EMAIL="saxenaakansh29@gmail.com"
' HEAD
