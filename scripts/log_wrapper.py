# log_wrapper.py

import logging
import sys
import pathlib

class IOWrapper:
    def logger_setup(self, log_directory=None):
        if log_directory is None:
            cp_dir = pathlib.Path.home().joinpath('.cellpose')
        else:
            cp_dir = pathlib.Path(log_directory)

        cp_dir.mkdir(exist_ok=True)
        log_file = cp_dir.joinpath('run.log')
        try:
            log_file.unlink()
        except:
            print('creating new log file')

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        logger = logging.getLogger(__name__)
        logger.info(f'WRITING LOG OUTPUT TO {log_file}')

        return logger, log_file



# todo: implement custom log handler to intercept information from training to feed into matplot
# class CellposeIntercepingHandler(logging.Handler):
#
# cellpose_logger = logging.getLogger('__name__ from core.py')
# cellpose_logger.addHandler('custom handler impopemmntnation')