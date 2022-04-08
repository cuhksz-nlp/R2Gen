import time


def time_executed(start_time, process_name):
    end_time = time.time()
    print("%s ended: %s" % (process_name, time.localtime(end_time)))
    execution_time = end_time - start_time
    hours, rem = divmod(execution_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print((process_name, " executed for {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)))
