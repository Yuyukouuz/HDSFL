from flsnn_client import client_start
import sys

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

if __name__ == '__main__':
    print_hi('PyCharm')
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if (len(sys.argv)) <= 1:
        client_start()
    elif (len(sys.argv)) == 2:
        client_start(int(sys.argv[1]))
    elif (len(sys.argv)) == 3:
        client_start(int(sys.argv[1]), int(sys.argv[2]))