# _*_ coding: utf-8 _*_

# Define source file list
compiled_dir = 'Libs'
source_dir = 'Common'

# Add import path
import sys
sys.path.append(source_dir)

#
from py_compile import compile
from log import Log
import shutil
import os


# log tool
log = Log()

# Init dir
if not os.path.isdir(compiled_dir):
    os.mkdir(compiled_dir)

# Compile
failed = []
source_files = [os.path.join(source_dir, i) for i in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, i)) and os.path.splitext(i)[1] == '.py']
for source_file in source_files:
    filename = os.path.split(source_file)[1]
    basename = os.path.splitext(filename)[0]
    pyc_path = os.path.join(compiled_dir, '%s.pyc' % basename)
    try:
        compile(source_file, pyc_path, doraise=True)
        log.show_log(u'%s.pyc 编译成功！' % basename)
    except:
        failed.append(source_file)

# Failed
log.show_log('')
for i in failed:
    log.show_log(u'%s 编译失败！！！' % i)

# delete tmp dir
if os.path.isdir('__pycache__'):
    shutil.rmtree('__pycache__')
if os.path.isdir(os.path.join(source_dir, '__pycache__')):
    shutil.rmtree(os.path.join(source_dir, '__pycache__'))
input(u'按回车键关闭窗口')
