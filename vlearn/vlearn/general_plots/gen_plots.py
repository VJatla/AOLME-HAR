import matplotlib.pyplot as plt
import pdb

def line_plot(x, y, title, x_label, y_label):
    
    fig, ax = plt.subplots()
    
    ax.plot(x,y)
    ax.set_title(title,fontsize=33)
    ax.tick_params(axis='both', which='major', labelsize=18)
 
    ax.set_xlabel(x_label, fontsize=21)
    ax.set_ylabel(y_label,fontsize=21)

    return fig, ax

def scatter_plot(x, y, title, x_label, y_label):
    
    fig, ax = plt.subplots()
    
    ax.scatter(x,y)
    ax.set_title(title,fontsize=33)
    ax.tick_params(axis='both', which='major', labelsize=18)
 
    ax.set_xlabel(x_label, fontsize=21)
    ax.set_ylabel(y_label,fontsize=21)

    return fig, ax

def histogram_plot(x, nbins, title, x_label, y_label):
    fig, ax = plt.subplots()

    n, bins, patches = ax.hist(x, nbins)
    ax.set_title(title,fontsize=33)
    ax.tick_params(axis='both', which='major', labelsize=18)
 
    ax.set_xlabel(x_label, fontsize=21)
    ax.set_ylabel(y_label,fontsize=21)

    return fig, ax