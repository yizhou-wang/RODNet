# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

fig = plt.figure(figsize=(8, 8))

fp = FontProperties(fname=r"assets/fontawesome-free-5.12.0-desktop/otfs/solid-900.otf")
symbols = {
    'pedestrian': "\uf554",
    'cyclist': "\uf84a",
    'car': "\uf1b9",
}
