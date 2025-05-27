#!/bin/bash

# GitHub mirror configuration
GITHUB_REPO="git@github.com:UriNeri/rolypoly.git"
GITHUB_REMOTE="github"

# Function to ensure GitHub remote is configured
setup_github_remote() {
    if ! git remote | grep -q "^$GITHUB_REMOTE$"; then
        echo "Adding GitHub remote..."
        git remote add $GITHUB_REMOTE $GITHUB_REPO
    fi
}

# Function to push current branch to GitHub
push_to_github() {
    setup_github_remote
    current_branch=$(git branch --show-current)
    echo "Pushing $current_branch to GitHub..."
    git push $GITHUB_REMOTE $current_branch
}

# Function to pull from GitHub
pull_from_github() {
    setup_github_remote
    current_branch=$(git branch --show-current)
    echo "Pulling $current_branch from GitHub..."
    git pull $GITHUB_REMOTE $current_branch
}

# Main execution
case "$1" in
    "push")
        push_to_github
        ;;
    "pull")
        pull_from_github
        ;;
    *)
        echo "Usage: $0 {push|pull}"
        exit 1
        ;;
esac
