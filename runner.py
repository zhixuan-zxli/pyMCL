import argparse
import pickle
from math import ceil
from dataclasses import dataclass
from os import mkdir, path
from numpy import load as np_load
from colorama import Fore, Style

@dataclass
class SolverParameters:
    dt: float
    Te: float
    num_checkpoints: int = 0
    num_frames: int = 0
    spaceref: int = 0 # space refinement level
    timeref: int = 0 # time refinement level

class Runner:

    name: str
    parser: argparse.ArgumentParser
    args: argparse.Namespace
    solp: SolverParameters

    step: int

    def __init__(self, name: str, solp: SolverParameters) -> None:
        self.name = name
        self.solp = solp
        
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--vis", help="Visualize the computation.", action="store_true")
        self.parser.add_argument("--resume", help="Resume from previous computations.")
        self.parser.add_argument("--spaceref", type=int, help="Spatial refinement level", default=0)
        self.parser.add_argument("--timeref", type=int, help="Time refinement level")

    def prepare(self) -> None:
        # parse the args
        self.args = self.parser.parse_args()
        del self.parser
        if self.args.timeref is None:
            self.args.timeref = self.args.spaceref
        print(Fore.GREEN + "Runner: " + self.name + Style.RESET_ALL)
        # get the resume information
        if self.args.resume:
            self.step = int(self.args.resume)
            self.resume_file = np_load(self._get_filename(self.args.resume + ".npz"))
            print("Successfully resumed from step = " + self.args.resume)
        else:
            self.step = 0
            mkdir(self.name)
            print("Start from step = 0")
        # save the parameters to disk
        with open(self._get_filename("SolverParameters"), "wb") as f:
            pickle.dump(self.solp, f)
        # calculate the time stepping information
        self.solp.dt /= 2**self.args.timeref
        num_steps = self.solp.Te / self.solp.dt
        print("Time steps = {}".format(num_steps))
        if self.solp.num_checkpoints > 0:
            self.solp.stride_checkpoint = ceil(num_steps / self.solp.num_checkpoints)
        else:
            self.solp.stride_checkpoint = 99999999 # some very big number
        if self.solp.num_frames > 0:
            self.solp.stride_frame = ceil(num_steps / self.solp.num_frames)
        else:
            self.solp.stride_frame = 99999999 # some very big number
        

    def finish(self) -> None:
        print(Fore.GREEN + "Completed. " + Style.RESET_ALL)

    def pre_step(self) -> bool:
        raise NotImplementedError

    def main_step(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        self.prepare()
        while True:
            flag = self.pre_step()
            if flag:
                break
            self.main_step()
            self.step += 1
        self.finish()

    # =================================================================
    def _get_filename(self, postfix: str) -> str:
        return path.join(self.name, postfix)
