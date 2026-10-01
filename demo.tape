# Records demo.gif for the README. Run from the project root:  vhs demo.tape
# Asks the agent one real question (costs ~$0.10-0.20 per recording).

Output demo.gif

Set Shell "bash"
Set FontSize 16
Set Width 1400
Set Height 900
Set Padding 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms
Set Framerate 24
Set WaitTimeout 300s

# Setup, not recorded: a clean prompt and the project's virtualenv.
Hide
Type "source .venv/bin/activate && export PS1='$ ' && clear"
Enter
Sleep 1s
Show

Type "python agent.py"
Sleep 400ms
Enter
Wait /you>\s*$/
Sleep 800ms

Type "What are the themes of 'Witchcraft for Wayward Girls'?"
Sleep 600ms
Enter

# The agent is done when its prompt comes back on the last line.
Wait /you>\s*$/
Sleep 4s
