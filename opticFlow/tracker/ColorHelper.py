import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

class MplColorHelper:

    def __init__(self, cmap_name, start_val, stop_val):
        if stop_val > 255:
            stop_val = 255
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        rgb = list(self.scalarMap.to_rgba(val))[:3]
        ret = [int(i * 255) for i in rgb]
        return [int(rgb[0]*255),int(rgb[1]*255),int(rgb[2]*255)]