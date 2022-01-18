from problems.robot import RobotMaze
from rl.acm import ACM
from problems.pole_balancing import PoleBalancing
from problems.towers_of_hanoi import TowersOfHanoi
from problems.gambler import Gambler
import json


def main():
    with open("config.json") as f:
        config = json.load(f)
    acm = ACM(config)
    acm.fit(RobotMaze({}))

    from contextlib import redirect_stdout
    with open('trace.txt', 'w') as f:
        with redirect_stdout(f):
            for i in range(20):
                print("#########################")
                print("VISUALIZATION", i)
                print("#########################")
                acm.predict(RobotMaze({}, visualize=True))
    # acm.fit(TowersOfHanoi())
    # acm.fit(Gambler())


if __name__ == '__main__':
    main()
