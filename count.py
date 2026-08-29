with open("a.txt", "r") as F:
    counts = 0
    for line in F:
        if line[0].isdigit():
            counts += 1
    print(counts)