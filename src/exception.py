from src.logger import logging
import sys

def err_message_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script [{0}] line number [{1}] error message [{2}]".format(file_name, exc_tb.tb_lineno, str(error))

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = err_message_detail(error=error_message, error_detail = error_detail)

    def __str__(self):
        logging.error(self.error_message)
        return self.error_message