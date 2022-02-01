from problems.labyrinth import Labyrinth
from rl.core import ACM
import json


def main():
    with open("config.json") as f:
        config = json.load(f)
    acm = ACM(config)
    acm.fit(Labyrinth())
    # acm.fit(PoleBalancing(config["pole_balancing"]))
    # acm.fit(TowersOfHanoi())
    # acm.fit(Gambler())


if __name__ == '__main__':
    main()
