import numpy as np
import matplotlib.pyplot as plt 
import sys
import math

# TODO
# add save flag for plots
# TODO
# function that locates lines over a region, and labels them on the transmission plot
# option A: run one model for each molecule to locate lines
# option B: refer to line list

# for array2d: clicking on a pixel prints its coordinates and value in console
# https://stackoverflow.com/questions/5836560/color-values-in-imshow-for-matplotlib
class EventHandler():
    def __init__(self, img, fig, ax):
        fig.canvas.mpl_connect('button_press_event', self.onpress)
        self.ax = ax
        self.img = img

    def onpress(self, event):
        if event.inaxes!=self.ax:
            return
        xi, yi = (int(round(n)) for n in (event.xdata, event.ydata))
        # note: indices must be switched! I don't know why
        value = self.img.get_array()[yi,xi]
        #color = self.img.cmap(self.img.norm(value))
        print("("+str(xi)+","+str(yi)+") has value "+str(value))

# plots a 2d array as an image
def array2d(array,lowerPercentile=3.,upperPercentile=99.7,x=None,y=None,title=None):
    lower = np.percentile(array, lowerPercentile)
    upper = np.percentile(array, upperPercentile)
    img = plt.imshow(np.clip(array,lower,upper),interpolation='none')
    fig = plt.gcf()
    ax = plt.gca()
    handler = EventHandler(img, fig, ax)
    plt.gray()
    if (x != None) and (y != None):
        # keep points inside the image range
        # & is an elementwise and
        valid = np.where( (x>=0) & (y>=0) & (x<array.shape[0]) & (y<array.shape[1]) )
        # make scatter plot
        plt.scatter(x[valid],y[valid],color="red",s=10)
    if title:
        plt.title(title)
    plt.show()
    return 0

def array(array,array2=None,colors=None):
    if not colors:
        colors = ["black","green","blue","red","purple","orange","pink","brown"]
    x = xrange(len(array))
    plt.plot(x,array,colors[0],linewidth=0.9)
    if array2 != None:
        x2 = xrange(len(array2))
        plt.plot(x2,array2,colors[1],linewidth=0.9)
    plt.xlim(x[0],x[-1])
    plt.tight_layout()
    plt.show()
    return 0

def paths(x,y,y2=None,y3=None,y4=None,y5=None,y6=None,y7=None,y8=None,linewidth=0.9,colors=None,legend=None):
    if not colors:
        colors = ["black","green","blue","red","purple","orange","pink","brown"]
    plt.plot(x,y,colors[0],linewidth=linewidth)
    yMin = np.min(y)
    yMax = np.max(y)
    if y2 != None:
        plt.plot(x,y2,colors[1],linewidth=linewidth)
        yMin = np.minimum(yMin,np.min(y2))
        yMax = np.maximum(yMax,np.max(y2))
    if y3 != None:
        plt.plot(x,y3,colors[2],linewidth=linewidth)
        yMin = np.minimum(yMin,np.min(y3))
        yMax = np.maximum(yMax,np.max(y3))
    if y4 != None:
        plt.plot(x,y4,colors[3],linewidth=linewidth)
        yMin = np.minimum(yMin,np.min(y4))
        yMax = np.maximum(yMax,np.max(y4))
    if y5 != None:
        plt.plot(x,y5,colors[4],linewidth=linewidth)
        yMin = np.minimum(yMin,np.min(y5))
        yMax = np.maximum(yMax,np.max(y5))
    if y6 != None:
        plt.plot(x,y6,colors[5],linewidth=linewidth)
        yMin = np.minimum(yMin,np.min(y6))
        yMax = np.maximum(yMax,np.max(y6))
    if y7 != None:
        plt.plot(x,y7,colors[6],linewidth=linewidth)
        yMin = np.minimum(yMin,np.min(y7))
        yMax = np.maximum(yMax,np.max(y7))
    if y8 != None:
        plt.plot(x,y8,colors[7],linewidth=linewidth)
        yMin = np.minimum(yMin,np.min(y8))
        yMax = np.maximum(yMax,np.max(y8))
    # add a legend
    if legend:
        leg = plt.legend(tuple(legend),fontsize=9,loc='upper left',scatterpoints=1,markerscale=10)
    xMin = np.min(x)
    xMax = np.max(x)
    if not math.isnan(xMin) and not math.isnan(xMax):
        plt.xlim(xMin-0.01*xMax,xMax+0.01*xMax)
    if not math.isnan(yMin) and not math.isnan(yMax):
        plt.ylim(yMin-0.01*yMax,yMax+0.01*yMax)
    plt.tight_layout()
    plt.show()
    return 0

def points(scatter,scatter2=None,scatter3=None,scatter4=None,scatter5=None,scatter6=None,scatter7=None,scatter8=None,scatter9=None,f=None,f2=None,f3=None,f4=None,f5=None,f6=None,f7=None,f8=None,f9=None,pars=None,pars2=None,pars3=None,pars4=None,pars5=None,pars6=None,pars7=None,pars8=None,pars9=None,colors=None,title=None,pointSize=None,legend=None,deleteZeros=False):
    # if plot is prone to overly large boundaries from unwanted zero points, cut 'em out.
    # this feature was added due to zeroes produced from applying masks to an array
    if deleteZeros:
        if scatter:
            valid = np.where( (scatter[0]!=0) & (scatter[1]!=0) )
            scatter = (scatter[0][valid], scatter[1][valid])
        if scatter2:
            valid = np.where( (scatter2[0]!=0) & (scatter2[1]!=0) )
            scatter2 = (scatter2[0][valid], scatter2[1][valid])
        if scatter3:
            valid = np.where( (scatter3[0]!=0) & (scatter3[1]!=0) )
            scatter3 = (scatter3[0][valid], scatter3[1][valid])
        if scatter4:
            valid = np.where( (scatter4[0]!=0) & (scatter4[1]!=0) )
            scatter4 = (scatter4[0][valid], scatter4[1][valid])
        if scatter5:
            valid = np.where( (scatter5[0]!=0) & (scatter5[1]!=0) )
            scatter5 = (scatter5[0][valid], scatter5[1][valid])
        if scatter6:
            valid = np.where( (scatter6[0]!=0) & (scatter6[1]!=0) )
            scatter6 = (scatter6[0][valid], scatter6[1][valid])
        if scatter7:
            valid = np.where( (scatter7[0]!=0) & (scatter7[1]!=0) )
            scatter7 = (scatter7[0][valid], scatter7[1][valid])
        if scatter8:
            valid = np.where( (scatter8[0]!=0) & (scatter8[1]!=0) )
            scatter8 = (scatter8[0][valid], scatter8[1][valid])
        if scatter9:
            valid = np.where( (scatter9[0]!=0) & (scatter9[1]!=0) )
            scatter9 = (scatter9[0][valid], scatter9[1][valid])
    # specify your own colors, else subject to crappy defaults
    if not colors:
        colors = ["black","blue","red","purple","orange","teal","green","pink","brown"]
    # create spaced x array to plot function with
    if scatter:
        xMin = np.min(scatter[0])
        xMax = np.max(scatter[0])
        yMin = np.min(scatter[1])
        yMax = np.max(scatter[1])
    if scatter2:
        xMin = np.minimum(np.min(scatter2[0]),xMin)
        xMax = np.maximum(np.max(scatter2[0]),xMax)
        yMin = np.minimum(np.min(scatter2[1]),yMin)
        yMax = np.maximum(np.max(scatter2[1]),yMax)
    if scatter3:
        xMin = np.minimum(np.min(scatter3[0]),xMin)
        xMax = np.maximum(np.max(scatter3[0]),xMax)
        yMin = np.minimum(np.min(scatter3[1]),yMin)
        yMax = np.maximum(np.max(scatter3[1]),yMax)
    if scatter4:
        xMin = np.minimum(np.min(scatter4[0]),xMin)
        xMax = np.maximum(np.max(scatter4[0]),xMax)
        yMin = np.minimum(np.min(scatter4[1]),yMin)
        yMax = np.maximum(np.max(scatter4[1]),yMax)
    if scatter5:
        xMin = np.minimum(np.min(scatter5[0]),xMin)
        xMax = np.maximum(np.max(scatter5[0]),xMax)
        yMin = np.minimum(np.min(scatter5[1]),yMin)
        yMax = np.maximum(np.max(scatter5[1]),yMax)
    if scatter6:
        xMin = np.minimum(np.min(scatter6[0]),xMin)
        xMax = np.maximum(np.max(scatter6[0]),xMax)
        yMin = np.minimum(np.min(scatter6[1]),yMin)
        yMax = np.maximum(np.max(scatter6[1]),yMax)
    if scatter7:
        xMin = np.minimum(np.min(scatter7[0]),xMin)
        xMax = np.maximum(np.max(scatter7[0]),xMax)
        yMin = np.minimum(np.min(scatter7[1]),yMin)
        yMax = np.maximum(np.max(scatter7[1]),yMax)
    if scatter8:
        xMin = np.minimum(np.min(scatter8[0]),xMin)
        xMax = np.maximum(np.max(scatter8[0]),xMax)
        yMin = np.minimum(np.min(scatter8[1]),yMin)
        yMax = np.maximum(np.max(scatter8[1]),yMax)
    if scatter9:
        xMin = np.minimum(np.min(scatter9[0]),xMin)
        xMax = np.maximum(np.max(scatter9[0]),xMax)
        yMin = np.minimum(np.min(scatter9[1]),yMin)
        yMax = np.maximum(np.max(scatter9[1]),yMax)
    # input scatter plots
    if not pointSize:
        pointSize = 1
    plt.scatter(scatter[0],scatter[1],color=colors[0],s=pointSize)
    if scatter2:
        plt.scatter(scatter2[0],scatter2[1],color=colors[1],s=pointSize)
    if scatter3:
        plt.scatter(scatter3[0],scatter3[1],color=colors[2],s=pointSize)
    if scatter4:
        plt.scatter(scatter4[0],scatter4[1],color=colors[3],s=pointSize)
    if scatter5:
        plt.scatter(scatter5[0],scatter5[1],color=colors[4],s=pointSize)
    if scatter6:
        plt.scatter(scatter6[0],scatter6[1],color=colors[5],s=pointSize)
    if scatter7:
        plt.scatter(scatter7[0],scatter7[1],color=colors[6],s=pointSize)
    if scatter8:
        plt.scatter(scatter8[0],scatter8[1],color=colors[7],s=pointSize)
    if scatter9:
        plt.scatter(scatter9[0],scatter9[1],color=colors[8],s=pointSize)
    # add a legend
    if legend:
        leg = plt.legend(tuple(legend),fontsize=9,loc='upper left',scatterpoints=1,markerscale=10)
    # input function plots
    if f:
        xPoints = np.arange(xMin,xMax,5)
        plt.plot(xPoints,f(xPoints,*pars),color=colors[0])
    if f2:
        plt.plot(xPoints,f2(xPoints,*pars2),color=colors[0])
    if f3:
        plt.plot(xPoints,f3(xPoints,*pars3),color=colors[0])
    if f4:
        plt.plot(xPoints,f4(xPoints,*pars4),color=colors[0])
    if f5:
        plt.plot(xPoints,f5(xPoints,*pars5),color=colors[0])
    if f6:
        plt.plot(xPoints,f6(xPoints,*pars6),color=colors[0])
    if f7:
        plt.plot(xPoints,f7(xPoints,*pars7),color=colors[0])
    if f8:
        plt.plot(xPoints,f8(xPoints,*pars8),color=colors[0])
    if f9:
        plt.plot(xPoints,f9(xPoints,*pars9),color=colors[0])
    # fix up and display
    if title:
        plt.title(title)
    plt.xlim(xMin-0.01*xMax,xMax+0.01*xMax)
    plt.ylim(yMin-0.01*yMax,yMax+0.01*yMax)
    plt.tight_layout()
    plt.show()

    return 0

def bar(x,y,title=None,xlabel=None,ylabel=None,show=True,f=None,pars=None):
    # Histogram of the data
    width = (max(x)-min(x))/len(x)
    plt.bar((x[:]-width/2.), y, width=width, color='teal', edgecolor='white', ecolor='black')
    # X-axis limits
    plt.xlim(min(x), max(x))
    plt.xticks(x,rotation=70)
    if f:
        xPoints = np.arange(min(x),max(x),(max(x)-min(x))/300.)
        plt.plot(xPoints,f(xPoints,*pars),color="black")
    if title != None:
        plt.title(title)
    if xlabel != None:
        plt.xlabel(xlabel)
    if ylabel != None:
        plt.ylabel(ylabel)
    plt.show()
    return 0

def getXYpoints(filename):
    x = []
    y = []
    with open(filename) as f:
        for line in f:
            xPt, yPt = line.split()
            xPt = str(xPt)
            yPt = str(yPt)
            x.append(xPt)
            y.append(yPt)
    x = np.array(x)
    y = np.array(y)
    x = x.astype(float)
    y = y.astype(float)
    return x,y

def getLBLRTMoutput(filelocation):
    # Pull data from text file 1
    f = open(filelocation,'r')
    lines=f.readlines()

    # identify beginning of data in file
    i = 0
    data = []
    copy=False
    while i<len(lines):
        if ("WAVENUMBER" in lines[i]): # the data starts after this line
            xTitle = "Wavenumber"
            if ("RADIANCE" in lines[i]):
                yTitle = "Radiance"
            elif ("TRANSMISSION" in lines[i]):
                yTitle = "Transmission"
            elif ("TEMPERATURE" in lines[i]):
                yTitle = "Temperature"
            elif ("OPTICAL DEPTH" in lines[i]):
                yTitle = "Optical Depth"
            else:
                yTitle = "unknown"
            copy = True
            i+=1
        if copy:
            data.append(lines[i]) 
        i+=1

    x = []
    y = []
    for item in data: # data points are formatted x y in lines
        try: # copy lines of data
            a = float(item[5:18]) #that line's wavenumber
            b = float(item[26:41]) #the corresponding radiance value
            x.append(a)
            y.append(b)
        except ValueError: # some lines are blank
            pass

    return x, y, xTitle, yTitle

def getATLASdata(f):
    f = open(f,'r')
    lines=f.readlines()

    # identify beginning of data in file
    i = 0
    data = []
    copy=False
    while i<len(lines):
        if ("WAVENUMBER" in lines[i]): # the data starts after this line
            xTitle = "Wavenumber"
            if ("RADIANCE" in lines[i]):
                yTitle = "Radiance"
            if ("TRANSMISSION" in lines[i]):
                yTitle = "Transmission"
            if ("TEMPERATURE" in lines[i]):
                yTitle = "Temperature"
            copy = True
            i+=1
        if copy:
            data.append(lines[i]) 
        i+=1

    x = []
    y = []
    for item in data: # data points are formatted x y in lines
        try: # copy lines of data
            a = float(item[5:18]) #that line's wavenumber
            b = float(item[26:41]) #the corresponding radiance value
            x.append(a)
            y.append(b)
        except ValueError: # some lines are blank
            pass

    n, x, y = np.loadtxt(f)
    return x, y, xTitle, yTitle

def files(fileNames):
    i = 0
    color = ['r','g','b','y','m']
    for f in fileNames:
        if ("TAPE30" in f) or ("TAPE29" in f) or ("TAPE28" in f) or ("TAPE27" in f):
            x, y, xTitle, yTitle = getLBLRTMoutput(f)
        elif ("atran" in f):
            x, y, xTitle, yTitle = getATLASdata(f)
        else:
            x, y = np.loadtxt(f, unpack=True)
            xTitle = 'x'
            yTitle = 'y'
        plt.plot(x, y, color[i], linewidth=0.9)
        i+=1
    plt.xlim(x[0],x[-1])
    plt.xlabel(xTitle)
    plt.ylabel(yTitle)
    plt.legend(tuple(fileNames),fontsize=9)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    if (len(sys.argv) == 7):
        files([ sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6] ])
    if (len(sys.argv) == 6):
        files([ sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5] ])
    elif (len(sys.argv) == 5):
        files([ sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4] ])
    elif (len(sys.argv) == 4):
        files([ sys.argv[1],sys.argv[2],sys.argv[3] ])
    elif (len(sys.argv) == 3):
        files([ sys.argv[1],sys.argv[2] ])
    else:
        files([ sys.argv[1] ])

