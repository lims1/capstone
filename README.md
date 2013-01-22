Capstone Spring 2013
====================

# Installing Git

Besides installing git, you'll need to set up SSH keys in order to push/pull from the remote github repository. Here's the link for that:

https://help.github.com/articles/generating-ssh-keys#platform-all

Make sure you pick the operating system you're using. Windows might be the trickiest; you'll probably have to install Cygwin.

Once your SSH keys are set up and github recognizes them, you'll be able to issue all of the following commands (and more) to your heart's content.

# Basics of Git

Here are a few commands that should get you up and running with github.

## git clone git@github.com:magsol/capstone.git

This command "clones" the repository into your local machine, automatically creating a folder called "capstone" in your current directory. You also have the option of creating a brand new repository on your machine, and then linking it to a remote repository (e.g. on github), but using clone is probably the easiest.

## git status

Issue this at anytime to see the status of your local repository. It will list files that have been changed from the last commit, as well as files that are in your repository but which have not yet been added to version control. We'll explore these concepts below.

## git add some/new/file.txt

When you create a new file in your repository, initially it is not under version control and will not appear in any commits you make. You have to explicitly place it under git control with this command.

## git commit -am "Commit message."

Once you've done some editing to the codebase, you'll want to commit these changes to the repository. This is where the distinction between the local repository on your machine and the remote repository (e.g. on github) becomes important: this commits the changes to your *local* repository, with the accompanying message in quotes (usually you want this to be a description of what changes you're committing: new features, bug fixes, refactoring, etc). Any modified files or new files you've added with "git add" will be committed.

## git push

This is how you sync your local repository with the remote one (e.g. github). When you've made one or more commits to your local repository, you can push these commits out to the remote repository with this command. All the commits will show up on the github web page.

## git pull

The whole point of versioning systems like git is so you can have multiple people working on the same codebase at the same time. From time to time you'll want to issue this command: it pulls down from the remote repository any changes others have made to the remote repository.

## Branching and merging

This is some advanced stuff that you won't need just yet but which is still incredibly useful. We'll go over this in detail later, but it allows multiple people to implement huge changes to the codebase without affecting the efforts of everyone else. In effect, branching allows you to "fork" off the codebase and implement any edits you want while others continue to commit to the "main" branch. You can even switch between branches to work on different versions of the same codebase simultaneously. Eventually, you'll want to merge the branches back together, which is a non-trivial task, but git makes it tractable. Here's a useful link: http://git-scm.com/book/en/Git-Branching-Basic-Branching-and-Merging

## Useful commands

`git branch -d <branchName>` : Deletes a local branch.

`git push origin --delete <branchName>` : Deletes a branch on github.
