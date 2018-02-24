from difflib import Differ

with open('practice03.py') as f1, open('demo.py') as f2:
    differ = Differ()

    for line in differ.compare(f1.readlines(), f2.readlines()):
        if line.startswith(" "):
            print(line[2:], end="")
