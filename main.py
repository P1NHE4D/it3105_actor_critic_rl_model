from rl.acm import ACM
from problems.pole_balancing import PoleBalancing
from problems.towers_of_hanoi import TowersOfHanoi
from problems.gambler import Gambler
import json


def main():
    with open("config.json") as f:
        config = json.load(f)
    acm = ACM(config)
    acm.fit(PoleBalancing(config["pole_balancing"]))
    # acm.fit(TowersOfHanoi())
    # acm.fit(Gambler())


if __name__ == '__main__':
    main()
