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
    spaceref: int = 0 # space refinement level
    timeref: int = 0 # time refinement level

class Runner:

    args: argparse.Namespace
    solp: SolverParameters

    step: int

    def __init__(self, solp: SolverParameters) -> None:
        self.solp = solp
        
        self._parser = argparse.ArgumentParser()
        self._parser.add_argument("output_dir", help="Output directory")
        self._parser.add_argument("--vis", help="Visualize the computation", action="store_true")
        self._parser.add_argument("--resume", help="Resume from previous computations")
        self._parser.add_argument("--spaceref", type=int, help="Spatial refinement level", default=0)
        self._parser.add_argument("--timeref", type=int, help="Time refinement level")
        self._parser.add_argument("--checkpoint", type=int, help="Number of checkpoints")
        self._parser.add_argument("--frame", type=int, help="Number of output frames")

    def prepare(self) -> None:
        # parse the args
        self.args = self._parser.parse_args()
        del self._parser
        if self.args.timeref is None:
            self.args.timeref = self.args.spaceref
        print(Fore.GREEN + "Runner: " + self.args.output_dir + Style.RESET_ALL)
        # get the resume information
        if self.args.resume:
            self.step = int(self.args.resume)
            self.resume_file = np_load(self._get_output_name(self.args.resume + ".npz"))
            print("Successfully resumed from step = " + self.args.resume)
        else:
            self.step = 0
            mkdir(self.args.output_dir)
            print("Start from step = 0")
        # save the parameters to disk
        with open(self._get_output_name("SolverParameters"), "wb") as f:
            pickle.dump(self.solp, f)
        # calculate the time stepping information
        self.solp.dt /= 2**self.args.timeref
        self.num_steps = ceil(self.solp.Te / self.solp.dt)
        print("Time steps = {}".format(self.num_steps))
        self.solp.stride_checkpoint = ceil(self.num_steps / self.args.checkpoint) if self.args.checkpoint else 99999999
        self.solp.stride_frame = ceil(self.num_steps / self.args.frame) if self.args.frame else 99999999
        

    def finish(self) -> None:
        print(Fore.GREEN + "\nCompleted. " + Style.RESET_ALL)

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
    def _get_output_name(self, postfix: str) -> str:
        return path.join(self.args.output_dir, postfix)
