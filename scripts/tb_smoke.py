import time

from rheidos.compute.profiler.tb import TBConfig, make_writer


def main() -> None:
    writer = make_writer(TBConfig(logdir="./_tb_logs/smoke"))
    step = 0
    while True:
        writer.add_scalar("smoke/counter", step, step)
        step += 1
        time.sleep(0.5)


if __name__ == "__main__":
    main()
