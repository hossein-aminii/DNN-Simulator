import os

cwd = os.getcwd()
files_and_dirs = os.listdir(path=cwd)
dirs = []
for p in files_and_dirs:
    if os.path.isdir(p):
        dirs.append(p)

for dir in dirs:
    files = os.listdir(dir)
    if len(files) == 1:
        print(f"directory {dir} contains only one file!")
