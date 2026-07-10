# Git Branching Guide

A concise guide to branching, merging, and collaboration workflows in Git.

## What a branch is

A Git branch is simply a movable pointer to a commit. Creating a branch is instantaneous
and costs a few bytes, because Git does not copy files: it just writes a new reference.
The default branch is commonly named `main`. The special pointer `HEAD` marks the branch
you currently have checked out.

## Creating and switching branches

`git branch feature-x` creates a branch, and `git switch feature-x` moves `HEAD` to it.
The shortcut `git switch -c feature-x` does both in one step. Work committed on
`feature-x` does not affect `main` until you merge it back.

## Merge vs. rebase

There are two ways to integrate changes from one branch into another:

- **Merge** (`git merge feature-x`) creates a *merge commit* with two parents, preserving
  the exact history of both branches. If `main` has not advanced since the branch was
  created, Git performs a *fast-forward* instead, simply moving the pointer.
- **Rebase** (`git rebase main`) rewrites the feature branch by replaying its commits on
  top of `main`, producing a linear history without merge commits.

The trade-off: merge preserves true history but can make the log harder to read; rebase
yields a clean linear log but rewrites commit hashes. The golden rule of rebasing is to
never rebase commits that have already been pushed to a shared branch, because rewriting
published history forces every collaborator to repair their local copies.

## Resolving conflicts

A conflict occurs when two branches modify the same lines of the same file. Git marks the
conflicting region with `<<<<<<<`, `=======`, and `>>>>>>>` markers, and stops for you to
edit the file. After choosing the correct content, `git add` the file and complete the
operation with `git merge --continue` or `git rebase --continue`.

## Common workflows

- **GitHub Flow**: a single long-lived `main` branch; every change starts as a short-lived
  feature branch and lands through a pull request with review and CI. Simple and suited
  to continuous deployment.
- **Git Flow**: adds long-lived `develop` and `release` branches plus `hotfix` branches.
  It fits products with scheduled, versioned releases, but adds ceremony that most web
  services do not need.
- **Trunk-based development**: everyone integrates into `main` at least daily, often
  behind feature flags. It minimizes merge pain at the cost of requiring strong automated
  testing.

## Useful inspection commands

`git log --oneline --graph` draws the commit graph in the terminal. `git diff main..feature-x`
shows what a branch adds. `git stash` shelves uncommitted changes so you can switch
branches with a clean working tree, and `git stash pop` restores them.
