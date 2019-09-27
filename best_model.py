import sys, re
lines = []
for line in sys.stdin:
    lines.append(line)
lines.sort(key=lambda x: float(re.sub("(.*_)|(\.pt)","",x)))
print(lines[0])
