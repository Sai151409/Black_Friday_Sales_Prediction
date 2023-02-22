import os
import sys

class BlackFridayException(Exception):
    def __init__(self, error_message : Exception, error_details : sys):
        super().__init__(error_message)
        self.error_message = BlackFridayException.get_detailed_error_message(
            error_message=error_message,
            error_details=error_details
        )
        
    @staticmethod
    def get_detailed_error_message(error_message : Exception, error_details : sys):
        _, _, exc_tab = error_details.exc_info()
        try_block_line_number = exc_tab.tb_lineno
        except_block_line_number = exc_tab.tb_frame.f_lineno
        filename = exc_tab.tb_frame.f_code.co_filename
        
        error_message = f"""
        Error has occured in the script : 
        [{filename}]
        try block line no : [{try_block_line_number}] and 
        expcept block line number : [{except_block_line_number}]
        message : [{error_message}]"""
         
        return error_message
    
    
    def __str__(self) -> str:
        return self.error_message
    
    def __repr__(self) -> str:
        return BlackFridayException.__name__.str()