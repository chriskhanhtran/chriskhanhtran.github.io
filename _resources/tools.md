---
title: "Tools"
toc: true
toc_label: "Table of Contents"
toc_icon: "bookmark"
excerpt: "My most frequently-used tools in Machine Learning projects."
header:
  teaser: "https://images.unsplash.com/photo-1534190239940-9ba8944ea261?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1489&q=80"
---
My most frequently-used tools to set up working space, monitor machine learning projects and deploy models.

## 1. Set Up Virtual Machines
- [Google Cloud setup and tutorial](https://github.com/cs231n/gcloud/): A comprehensive tutorial to set up a Google Cloud virtual machine by [CS231n](http://cs231n.stanford.edu/) (Stanford).
- [Server setup tutorial by fast.ai](https://course.fast.ai/start_gcp.html): Tutorials to set up virtual machines on Google Cloud, Azure, AWS... for fast.ai courses.
- [Remote SSH with Visual Studio Code](https://code.visualstudio.com/blogs/2019/07/25/remote-ssh): A tutorial to connect to virtual machines on VSCode with Remote SSH. It helps making working on VM easy and smooth as working locally.

## 2. Set Up Work Spaces
- [Byobu](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-byobu-for-terminal-management-on-ubuntu-16-04) for Terminal Management: My favorite tool when wokring on cloud. Byobu allows us to create sessions to open multiple terminal windows. We can detach a session and turn off our computer when training models on cloud and reactivate the session later to continue the training process.
- [Managing Anaconda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html): Everything we need to know to manage our conda virtual environments.
- [Setting up a Data Science environment using WSL and Jupyter](https://towardsdatascience.com/setting-up-a-data-science-environment-using-windows-subsystem-for-linux-wsl-c4b390803dd): A guide to set up Windows Subsystem for Linux (WSL), install Anaconda and Jupyter on Ubuntu.
- [Oh-my-zsh and agnoster theme](https://blog.joaograssi.com/windows-subsystem-for-linux-with-oh-my-zsh-conemu/): Set up a beautiful and functional theme for your terminal.
![](https://github.com/apodkutin/agnoster-zsh-theme/raw/customize-prompt/agnoster_customization.gif)
- [Setting up development environment for machine learning](https://www.youtube.com/watch?v=N9lo_UxSkWA) by Abhishek Thakur: A YouTube tutorial to set up development environment for machine learning projects.
- [Ipykernel](https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments): Create kernels for different environments so we can switch environments within Jupyter Lab/Notebook.

## 3. VSCode
![](/assets/images/blogs/vscode-workspace.gif)
My favorite VSCode setup for machine learning projects 🤗. First, I use `RemoteSSH` extension to SSH to a virtual machine. It makes interacting with files on cloud as smooth as working locally. Then I use `Byobu` to open multiple terminal windows: `htop` (monitor CPUs), `watch -n1 nvidia-smi` (monitor GPUs), `jupyter lab`, `tensorboard dev upload --logdir .` (upload logs to [TensorBoard.dev](https://tensorboard.dev/)) etc. I can press `F2` to create a new window and press `F4` to switch between windows. Extremely convenient ⚡!
{: .small}

- My favorite theme: Dracula 🧛‍♂️
- Extensions:
  - 🌟RemoteSSH: Open any folder on a remote machine using SSH and take advantage of VS Code's full feature set.
  - GitLens: Make Git more powerful on Visual Studio Code.
  - [Peacock](https://www.peacockcode.dev/): Subtly change the workspace color of our workspace. Ideal when we have multiple VSCode instances.
  - vscode-icons: Icons for Visual Studio Code.
  - Bracket Pair Colorizer: A customizable extension for colorizing matching brackets.
  - Code Spell Checker: Spelling checker for source code.
  - Setting Sync: Synchronize settings, snippets, themes, file icons, launch, keybindings, workspaces and extensions across multiple machines.
  - Prettier: Code formatter using prettier.
- [Config code formatter](https://code.visualstudio.com/docs/python/editing#_formatting)
- [Add vertical rulers](https://stackoverflow.com/questions/29968499/vertical-rulers-in-visual-studio-code)

## 4. Git
Some basic guides to interact with Git.
- [GitHub Guides](https://guides.github.com/)
- [Adding an existing project to GitHub using the command line](https://docs.github.com/en/github/importing-your-projects-to-github/adding-an-existing-project-to-github-using-the-command-line)
- [Oh-my-zsh Git alias](https://github.com/ohmyzsh/ohmyzsh/wiki/Cheatsheet): for extremely fast Git actions.

## 5. Production
After finishing important machine learning projects, I always want to deploy a simple prototype of my model with Streamlit to illustrate its usage or to present my works with the audience.
- [Streamlit API](https://docs.streamlit.io/en/stable/api.html#magic-commands)
- [Quickly Build and Deploy a Dashboard with Streamlit and Heroku](https://towardsdatascience.com/quickly-build-and-deploy-an-application-with-streamlit-988ca08c7e83)

## 6. Cheatsheet
- [Linux](https://files.fosswire.com/2007/08/fwunixref.pdf)
- [Regular Expression](https://www.programiz.com/python-programming/regex)
- [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#links)

## 7. Misc
- Docstring: [Example Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- LaTex:
  - [LaTex Online Editor](https://latex.codecogs.com/eqneditor/editor.php)
  - [Handwritten symbols to LaTex](http://write-math.com/)
  - [Math symbols collection](https://leimao.github.io/downloads/tools/Latex-Guidance/Symbols.pdf)
- Style guide: [Computer Science Style Guide: ACM, APA and IEEE](https://dal.ca.libguides.com/c.php?g=257109&p=1717772#jaxiee)
- [Google Drive Link Generator](https://www.wonderplugin.com/online-tools/google-drive-direct-link-generator/)
