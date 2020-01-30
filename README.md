# Sphere Decoding
In this project, I have implemented the sphere decoding algorithm based on [this](http://users.ece.utexas.edu/~hvikalo/pubs/paper1r.pdf) paper(SDIRS algorithm), and after creating about 10,000 Test data, I have used the Deep Deural Network to learn the radius of the algorithm(DLbased_SD).



## Technology
**Python** and using **Keras** for Deep Neural Net.
also, I have used **Jupyter notebook** in DNN codes to a better visual understanding


## Results

### Part I) sphere decoding algorihtm:

number of Floating operation points for sphere decoding in infinite lattice case with noise variance = 0.01, over M (number of antennas):[SphereDecodingAlgo.py]

![](https://github.com/SINAABBASI/Sphere-Decoding/blob/master/Results/infiniteLattice_0.01.png)

finite lattice case for 16-QAM symbols at snr = 20dB :[finite16QAM.py]

![](https://github.com/SINAABBASI/Sphere-Decoding/blob/master/Results/16QAM_20db.png)


Comparing 4-QAM and 16-QAM symbols at snr = 20dB:[16QAM_4QAM.py](https://github.com/SINAABBASI/Sphere-Decoding/blob/master/16QAM_4QAM.py)

![](https://github.com/SINAABBASI/Sphere-Decoding/blob/master/Results/16QAMvs4QAM_20db.png)

Comparing (4,16,64,256)-QAM through 10dB-40dB:[10dB_to_40dB.py](https://github.com/SINAABBASI/Sphere-Decoding/blob/master/10dB%20_to_40dB.py)

![](https://github.com/SINAABBASI/Sphere-Decoding/blob/master/Results/10dB_to_40dB.png)

### Part II) Deep Learning part (Comparing DLbased_SD and SDIRS):
#### source codes: 
1)Deep neural network for learning: [DNN/DNN.ipynb](https://github.com/SINAABBASI/Sphere-Decoding/blob/master/DNN/DNN.ipynb)

2)Comparing the DLbased_SD and SDIRS: [Compare_DNN_SDIRS/comp_DNNandSDIRS.py](https://github.com/SINAABBASI/Sphere-Decoding/blob/master/Compare_DNN_SDIRS/comp_DNNandSDIRS.py)

Comparing the average of runnig time :( Time average of DLbased_SD / Time average of SDIRS):

![](https://github.com/SINAABBASI/Sphere-Decoding/blob/master/Results/DNNvsSDIRS_Time.png)

Comparing number of lattice points inside the sphere as important parameter for complexity:

![](https://github.com/SINAABBASI/Sphere-Decoding/blob/master/Results/DNNvsSDIRS_NumberOfLatticePoints.png)
