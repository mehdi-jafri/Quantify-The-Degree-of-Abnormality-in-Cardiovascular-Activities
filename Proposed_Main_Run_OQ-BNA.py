import tkinter as tk
import tkinter.ttk as ttk
from tkinter import *
from tkinter import messagebox
import sys
import os

root = tk.Tk()
root.title("")
root.geometry('1350x768+0+0')

tk.Label(root, 
		 text="OUTLIER QUANTIFICATION USING BATCH NORMALIZED AUTOENCODER FOR  ",
		 fg = "light green",
		 bg = "dark green",
		 font = "Helvetica 16 bold italic").pack(pady=30,padx=0)
		 
tk.Label(root, 
		 text="CONTINUOUS MONITORING OF CARDIOVASCULAR ACTIVITIES USING PPG ",
		 fg = "light green",
		 bg = "dark green",
		 font = "Helvetica 16 bold italic").pack(pady=1,padx=0)
		 
def b1():   
    
    os.system('python Plot_PPG_Signals.py')

def b2():

    os.system('python Quasi_residue_Empirical_Gradient_Filter_Preprocessing.py')    

def b3():
    
    os.system('python Fractal_Dimension_Time_Domain_feature_extraction.py')
    os.system('python Product-to-sum_Fourier_Frequency_Domain_feature_extraction.py')

def b31():    

    os.system('python Batch_Normalized_Auto_encoder_Threshold_Anomaly_Detection_Quantifier.py')

def b4():
    
    os.system('python MAE_outlier_detection.py')   
	
def b5():    
    
    os.system('python Precision_Recall_Accuracy.py')

def b6():
    
    os.system('python F1_Score.py')  


b1=Button(root,text="MIMIC-IV Waveform Database",command=b1,bg="black",fg="white",font = "Helvetica 13 bold italic")

b1.place(x=200,y=200)

b1.configure(width=55,height=1)


b2=Button(root,text="Quasi-residue Empirical Gradient Filter-based Pre-processing ",command=b2,bg="black",fg="white",font = "Helvetica 13 bold italic")

b2.place(x=200,y=250)

b2.configure(width=55,height=1)


b3=Button(root,text="Fractal Dimension_TD_Product-to-sum Fourier_FD_Feature Extraction",command=b3,bg="black",fg="white",font = "Helvetica 13 bold italic")

b3.place(x=200,y=300)

b3.configure(width=55,height=1)


b31=Button(root,text="Batch Normalized Auto-encoder Threshold_Outlier Detection Quantifier ",command=b31,bg="black",fg="white",font = "Helvetica 13 bold italic")

b31.place(x=200,y=350)

b31.configure(width=55,height=1)


l2=tk.Label(root,text="Performance",fg = "light green",bg = "dark green",font = "Helvetica 16 bold italic")
l2.place(x=980,y=400)

b4=Button(root,text=" MAE of outlier detection (%)",command=b4,bg="black",fg="white",font = "Helvetica 13 bold italic")
b4.place(x=800,y=450)
b4.configure(width=45,height=1)

b5=Button(root,text="Precision, Recall and Accuracy",command=b5,bg="black",fg="white",font = "Helvetica 13 bold italic")
b5.place(x=800,y=500)
b5.configure(width=45,height=1)

b6=Button(root,text="F1-score",command=b6,bg="black",fg="white",font = "Helvetica 13 bold italic")
b6.place(x=800,y=550)
b6.configure(width=45,height=1)

tk.Label(root, 
		 text="Mir Mehdi Ali Jafri, Research Scholar, ORCID- 0000-0003-1305-1651",
		 fg = "light green",
		 bg = "dark green",
		 font = "Helvetica 16 bold italic").pack(pady=1,padx=0)
tk.Label(root, 
		 text="Contact: mirmehdialijafri@gmail.com, Mobile/WhatsApp: +91-9960339321",
		 fg = "light green",
		 bg = "dark green",
		 font = "Helvetica 16 bold italic").pack(pady=1,padx=0)


root.mainloop()
