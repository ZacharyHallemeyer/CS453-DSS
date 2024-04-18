import random


def main(rangeStart, rangeEnd, N, filePath):
    data = ""

    for i in range(N):
        data += f"{random.randint(rangeStart, rangeEnd)},{random.randint(rangeStart, rangeEnd)}\n"
    data = data[:-1]

    with open(filePath, "w") as f:
        f.write(data)

main(0, 10000, 10000000, "xy10000000.txt")
