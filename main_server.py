
from flsnn_server import server_start
import os

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

if __name__ == '__main__':
    print_hi('PyCharm')
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    server_start()