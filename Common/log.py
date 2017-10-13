# _*_ coding: utf-8 _*_
from __future__ import print_function
import inspect
import time


class Log:
    def __init__(self, need_show_log=True, log_file_path='./log.txt'):
        #need_show_log = False
        self.__need_show_log = need_show_log
        self.__log_file_path = log_file_path

    def set_log_visible(self, visible):
        self.__need_show_log = visible
    
    def get_current_time(self):
        return time.strftime('%H:%M:%S', time.localtime())

    def show_var(self, *vars):
        if not self.__need_show_log:
            return

        finfo = inspect.getframeinfo(inspect.currentframe().f_back)[3][0].strip()
        var_with_sep = finfo[finfo.find('(')+1 : finfo.rfind(')')]
        var_items = [item.strip() for item in var_with_sep.split(',')]
        index = 0
        for var in vars:
            print('%s:' % var_items[index], var)
            index += 1

    def show_log(self, log_msg):
        if not self.__need_show_log:
            return
        print(log_msg)
    
    def show_log_start_with_current_time(self, log_msg):
        if not self.__need_show_log:
            return
        print('%s: %s' % (self.get_current_time(), log_msg))

    def write_log(self, log_msg):
        if not self.__need_show_log:
            return
        with open(self.__log_file_path, 'a') as f:
            f.write(log_msg + '\n')

    def write_log_start_with_current_time(self, log_msg):
        if not self.__need_show_log:
            return
        with open(self.__log_file_path, 'a') as f:
            f.write('%s: %s' % (self.get_current_time(), log_msg))

    def show_and_write_log(self, log_msg):
        if not self.__need_show_log:
            return
        self.show_log(log_msg)
        self.write_log(log_msg)

    def show_and_write_log_start_with_current_time(self, log_msg):
        if not self.__need_show_log:
            return
        self.show_log_start_with_current_time(log_msg)
        self.write_log_start_with_current_time(log_msg)
