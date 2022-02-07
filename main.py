from problems.gambler import Gambler
from problems.labyrinth import Labyrinth
from rl.core import ACM
import json


def main():
    with open("config.json") as f:
        config = json.load(f)
    acm = ACM(config)
    acm.fit(Gambler({"win_prob": 0.4}))
    # acm.fit(PoleBalancing(config["pole_balancing"]))
    # acm.fit(TowersOfHanoi())
    # acm.fit(Gambler())


if __name__ == '__main__':
    main()
