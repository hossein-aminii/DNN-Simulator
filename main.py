from config import Config
import time
from simulator import SimulatorDispatcher
from results.utils import TrainResultsToExcel


def main():

    config = Config()
    dispatcher = SimulatorDispatcher(config=config)
    simulator = dispatcher.dispatch()
    simulator.run()


if __name__ == "__main__":
    # start chronometer
    start = time.time()
    # ---------------------------------------------
    # run simulator
    main()
    # ---------------------------------------------
    # stop chronometer
    end = time.time()
    # ----------------------------------------------
    # calculate running time
    take_long = end - start
    hours = int(take_long // 3600)
    take_long = take_long % 3600
    minutes = int(take_long // 60)
    take_long = take_long % 60
    seconds = int(take_long)
    # ------------------------------------------------
    # show running time
    print("this simulation take {:02d}:{:02d}:{:02d} long!".format(hours, minutes, seconds))

    # json_filepath = "results\\models\\train_results.json"
    # excel_filepath = "results\\excel_files\\model_accuracy.xlsx"
    # results_to_excel = TrainResultsToExcel(
    #     json_filepath=json_filepath, excel_filepath=excel_filepath, worksheet_name="Base-Model-IMDB"
    # )
    # results_to_excel.convert()
