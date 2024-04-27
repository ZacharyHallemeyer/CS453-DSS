import argparse

from math import sqrt


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str)
parser.add_argument("-e", "--epsilon", type=float)
parser.add_argument("-q", "--query", default=None, type=float, nargs="*")

args = parser.parse_args()


def query(points, q, e):
    count = 0
    for p in points:
        dist = sqrt(
            ((q[0] - p[0]) * (q[0] - p[0])) + ((q[1] - p[1]) * (q[1] - p[1]))
        )
        if dist <= e:
            count += 1
    return count


if __name__ == "__main__":
    points = []
    with open(args.dataset, "r") as fh:
        for line in fh.readlines():
            line = line.replace("\n", "").split(",")
            points.append([float(line[0]), float(line[1])])
    if args.query:
        count = query(points, args.query, args.epsilon)
        print(
            f"\nFor point ({args.query[0]}, {args.query[1]}), the number of points"
            f" within epsilon = {args.epsilon} is {count}"
        )
    else:
        count = 0
        for p in points:
            count += query(points, p, args.epsilon)
        print(f"\nTotal number of neighbors in epsilon = {count}")
