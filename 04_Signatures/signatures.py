#!/usr/bin/env python3
import numpy as np
import fastdtw
import sys
from pathlib import Path
from collections import defaultdict

RADIUS = 10
FEATURES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
FEATURES_STR = ''.join(map(str, FEATURES))

def main():
    filename = "result-{}-{}.txt".format(RADIUS, FEATURES_STR)
    with open(filename, 'w') as f:
        print("Loading data", file=sys.stderr)
        users, enrollment, verification, truth = load_data()

        results = dict()
        for u, r in test(users, enrollment, verification):
            results[u] = r
            scores = ["{}, {}".format(id, score) for id, score in r.items()]
            scores = str.join(', ', scores)
            print("{}, {}".format(u, scores), file=f)

        print("Done!", file=sys.stderr)
        s = score(results, truth)
        print("Score={} \tRADIUS={} \tFEATURES={} \t"
              .format(s, RADIUS, FEATURES_STR), file=sys.stderr)
        print("Score={} \tRADIUS={} \tFEATURES={} \t"
              .format(s, RADIUS, FEATURES_STR))


def test(users, enrollment, sigs):
    for i, u in enumerate(users):
        print("Testing user {}/{}".format(i+1, len(users)), file=sys.stderr)
        yield u, test_user(enrollment[u], sigs[u])


def score(results, truth):
    from sklearn.metrics import roc_auc_score
    scores = []
    for u, res in results.items():
        ids = [*sorted(truth[u])]
        t = [truth[u][i] for i in ids]
        r = [res[i] for i in ids]
        scores += [roc_auc_score(t, r)]

    return np.mean(scores)


def load_data():
    root = Path("data")

    with (root / "users.txt").open() as f:
        users = [user.strip() for user in f]

    truth = defaultdict(dict)
    with (root / "gt.txt").open() as f:
        for line in f:
            id, t = line.strip().split(' ')
            user = id.split('-')[0]
            truth[user][id] = 1 - int(t == 'g')  # 0 = genuine, 1 = forgery
    truth = dict(truth)

    enrollment = load_sigs(root / "enrollment")
    verification = load_sigs(root / "verification")
    return users, enrollment, verification, truth


def load_sigs(path):
    sigs = defaultdict(dict)
    for path in path.iterdir():
        user, id, sig = load_sig(path)
        sigs[user][id] = sig

    return dict(sigs)


def load_sig(path):
    with path.open() as f:
        id = path.name.split('.')[0]
        user = id.split('-')[0]

        sig = [[float(x) for x in line.split(' ')] for line in f]
        sig = np.array(sig)
        sig = extract_features(sig)
        return user, id, sig


def extract_features(sig):
    t, x, y, pressure, penup, azimuth, inclin = sig.T

    # Re-center
    x -= np.mean(x)
    y -= np.mean(y)

    # Rescale to 32px width
    w = x.max() - x.min()
    k = 32 / w
    x *= k
    y *= k

    # Compute velocities and accelerations
    def diff(v, n=1):
        v = np.diff(x, n=n)
        return np.r_[[0]*n, v]

    dt = np.diff(t)
    dt = np.r_[dt[0], dt]

    vx = diff(x, 1) / dt
    ax = diff(x, 2) / dt
    vy = diff(y, 1) / dt
    ay = diff(y, 2) / dt

    # Return
    sig = [x, y, vx, vy, ax, ay, pressure, azimuth, inclin]
    sig = np.array(sig)
    return sig[FEATURES]


def test_user(enrollment, sigs):
    results = dict()
    for i, (id, sig) in enumerate(sigs.items()):
        print("  - Testing signature {}/{}".format(i+1, len(sigs)),
              file=sys.stderr)
        results[id] = test_sig(enrollment, sig)

    return results


def test_sig(enrollment, sig):
    return min(dtw(e, sig) for e in enrollment.values())


def dtw(a, b):
    return fastdtw.fastdtw(a.T, b.T, radius=RADIUS, dist=len(a))[0]


if __name__ == '__main__':
    if len(sys.argv) == 3:
        RADIUS = int(sys.argv[1])
        FEATURES = [*sorted(int(c) for c in sys.argv[2])]
        FEATURES_STR = ''.join(map(str, FEATURES))

        print("Running with \tRADIUS={} \tFEATURES={}"
              .format(RADIUS, FEATURES_STR), file=sys.stderr)
        main()
    else:
        print("Usage: {} RADIUS FEATURES".format(sys.argv[0]), file=sys.stderr)
        print("  Example: {} {} {}".format(sys.argv[0], RADIUS, FEATURES_STR),
              file=sys.stderr)
