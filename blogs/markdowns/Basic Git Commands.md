# Basic Git Commands

## Git clone

```
git clone <https://name-of-the-repository-link>
```

## Git branch

```bash
git branch <branch-name> # Create new branch locally
git push -u <remote> <branch-name> # Push new branch to remote repository
git branch or git branch --list # View branches
git branch -d <branch-name> # Delete branch
```

## Git checkout

```bash
git checkout <name-of-your-branch> # Switch to another branch
git checkout -b <name-of-your-branch> # Shortcut to create and switch to new branch
```

## Git status

```bash
git status # Gives information about curent branch, such as whether branch is up to date
```

## Git add

```bash
git add <file> # Include changes of a file into our next commmit
git add . # Include everything at once
git remote add origin <https://name-of-the-repository-link> #Connect local git repository to a remote repository and save repository link in variable origin
```

## Git commit

```bash
git commit -m "commit message" # Sets a checkpoint / new version **Commits saved locally
```

## Git push

```bash
git push <remote> <branch-name> # Send commits to remote repository
git push -u origin <branch_name> # -u allows to make shortcut for next commit to 'git push'
```

## Git pull

```bash
git pull <remote> # Get updates from remote repository (Combination of fetch + merge)
```

## Git revert

```bash
git log -- oneline # Prints out commit history
git revert <commit-id> # Undoes given commit, creates new commit without deleteing old one
```

## Git merge

```bash
git checkout dev
git fetch # Updates local dev branch
git merge <branch-name> # Merge new branch to dev
```

## Git diff

```bash
git diff main dev # Compare differences between main and dev branch in terminal
git difftool main dev # Compare differences between main and dev branch in diff tool
git difftool <commit hash code> <commit hash code> # Compare difference between two commits
```