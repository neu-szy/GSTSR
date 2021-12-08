from tensorboardX import SummaryWriter
import time
class TB:
    writer = None

    @staticmethod
    def create_writer(name):
        date = time.strftime("%Y-%m-%d %H:%M:%S")
        TB.writer = SummaryWriter(f"/home/zhiyi/pycharm-tmp/basicsr/.runs/{name}" + "_" + date)

    @staticmethod
    def add_scalar(name, tag, scalar_value, global_step=None, walltime=None):
        if TB.writer is None:
            TB.create_writer(name)
        TB.writer.add_scalar(tag, scalar_value, global_step, walltime)

if __name__ == "__main__":
    TB.add_scalar("testqwe", "123", 1234)