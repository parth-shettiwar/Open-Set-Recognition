import numpy as np
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy as sp
from sklearn.linear_model import orthogonal_mp_gram
from sklearn.preprocessing import normalize

csev=np.array([])
with open('newer.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csev)
csvFile.close()
