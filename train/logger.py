import time
import matplotlib.pyplot as plt
from hyperParams import hyp
import os

class logger:

    def __init__(self):
        folder = './log_directory' + '/' +  hyp["log_dir"]
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.fname = "log_{}.txt".format(time.time())
        self.handle = open(self.fname,'a')
        self.track_stats = {}
        self.current_epoch = -1

    def log(self,*args,**kwargs):
        print(*args,**kwargs)
        self.handle.write(*args,*kwargs)
        self.handle.write("\n")
        self.handle.flush()

    def close(self):
        self.handle.close()
        print("Log saved to {}".format(self.fname))


    def plot(self, data, label, epoch=-1):
        if epoch==-1:
            self.track_stats["global_{}".format(label)] = data
        else:
            self.current_epoch=epoch
            key="epoch{}_{}".format(epoch,label)
            self.track_stats[key]=data
        self.plot_stats()


    def plot_stats(self):
        for key in self.track_stats.keys():
            if "epoch" in key:
                plt.plot(self.track_stats[key],key)
                plt.savefig("epoch_stats.png")
                plt.close()
                
        for key in self.track_stats.keys():
            if "global" in key:
                plt.plot(self.track_stats[key],key)
                plt.savefig("global_stats.png")
                plt.close()
                
        for key in self.track_stats.keys():
            if "epoch{}".format(self.current_epoch) in key and self.current_epoch!=-1:
                plt.plot(self.track_stats[key],key)
                plt.savefig("epoch_{}.png".format(self.current_epoch))
                plt.close()
