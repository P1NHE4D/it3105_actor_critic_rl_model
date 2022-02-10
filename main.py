import sys

from problems.gambler import Gambler
from problems.labyrinth import Labyrinth
from problems.pole_balancing import PoleBalancing
from problems.towers_of_hanoi import TowersOfHanoi
from rl.core import ACM
import json


def main():
    config_file = 'default.json'
    if len(sys.argv) >= 2:
        config_file = sys.argv[1]
    with open(config_file) as f:
        config = json.load(f)

    if config['problem'] == 'labyrinth':
        problem = Labyrinth()
    elif config['problem'] == 'towers':
        problem = TowersOfHanoi(**config['problem_params'])
    elif config['problem'] == 'pole':
        problem = PoleBalancing(**config['problem_params'])
    elif config['problem'] == 'gambler':
        problem = Gambler(**config['problem_params'])
    else:
        raise Exception('Unknown problem')

    acm = ACM(config)
    acm.fit(problem)
    # acm.fit(PoleBalancing(config["pole_balancing"]))
    # acm.fit(TowersOfHanoi())
    # acm.fit(Gambler())


if __name__ == '__main__':
    main()
