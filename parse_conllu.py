import sys

path = sys.argv[1]

with open("conllu_pos.txt", "w") as out:
    with open(path, "r") as conll:
        for line in conll.read().strip().split("\n"):
            if line == '':
                out.write("\n")
                continue
            if line[0] == '#':
                continue
            parts = line.split("\t")
            out.write("%s\t%s,%s\n" % (parts[1], parts[3], parts[5]))
            # out.write("%s\t%s\n" % (parts[1], parts[3]))
