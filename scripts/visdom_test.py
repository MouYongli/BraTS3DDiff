import visdom
import numpy as np
import time

vis = visdom.Visdom(port=8097)
vis.text('Hello, world!',win='img1')
vis.image(np.ones((10, 10)),win='img',opts={'title':'img11','caption':'img12'})
time.sleep(3)

vis.image(np.zeros((3, 10, 10)),win='img',opts={'title':'img11','store_history':True})
