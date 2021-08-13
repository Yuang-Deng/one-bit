import numpy as np
import matplotlib.pyplot as plt  

stuap = {}
tcap = {}
metrix = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']

f = open("./output/log.txt")
line = f.readline()
while line:
    if 'eta' in line and (int(line.split('  ')[2].split(' ')[1]) + 21) % 1000 == 0:
        iter = int(line.split('  ')[2].split(' ')[1]) + 21
        stuap[str(iter)] = {}
        tcap[str(iter)] = {}
        while '|  AP50  |' not in line:
            line = f.readline()
        line = f.readline()
        line = f.readline()
        strs = line.split('|')
        for i in range(len(metrix)):
            stuap[str(iter)][metrix[i]] = strs[i + 1]
        while '|  AP50  |' not in line:
            line = f.readline()
        line = f.readline()
        line = f.readline()
        strs = line.split('|')
        for i in range(len(metrix)):
            tcap[str(iter)][metrix[i]] = strs[i + 1]
    line = f.readline()
f.close()

iter = stuap.keys()
sap = []
sap50 = []
sap75 = []
saps = []
sapm = []
sapl = []
tap = []
tap50 = []
tap75 = []
taps = []
tapm = []
tapl = []
for i in iter:
    sap.append(float(stuap[i]['AP'].strip()))
    sap50.append(float(stuap[i]['AP50'].strip()))
    sap75.append(float(stuap[i]['AP75'].strip()))
    saps.append(float(stuap[i]['APs'].strip())) 
    sapm.append(float(stuap[i]['APm'].strip())) 
    sapl.append(float(stuap[i]['APl'].strip())) 
    tap.append(float(tcap[i]['AP'].strip()))
    tap50.append(float(tcap[i]['AP50'].strip()))
    tap75.append(float(tcap[i]['AP75'].strip()))
    taps.append(float(tcap[i]['APs'].strip())) 
    tapm.append(float(tcap[i]['APm'].strip())) 
    tapl.append(float(tcap[i]['APl'].strip())) 
print(list(iter))

s1=plt.plot(iter,sap,label='sap', color ='b', marker='o')
s2=plt.plot(iter,sap50,label='sap50', color ='g', marker='o')
s3=plt.plot(iter,sap75,label='sap75', color ='r', marker='o')
s4=plt.plot(iter,saps,label='saps', color ='c', marker='o')
s5=plt.plot(iter,sapm,label='sapm', color ='m', marker='o')
s6=plt.plot(iter,sapl,label='sapl', color ='y', marker='o')
t1=plt.plot(iter,tap,label='tap', color ='b', marker='D')
t2=plt.plot(iter,tap50,label='tap50', color ='g', marker='D')
t3=plt.plot(iter,tap75,label='tap75', color ='r', marker='D')
t4=plt.plot(iter,taps,label='taps', color ='c', marker='D')
t5=plt.plot(iter,tapm,label='tapm', color ='m', marker='D')
t6=plt.plot(iter,tapl,label='tapl', color ='y', marker='D')
plt.title('The Lasers in Three Conditions')
plt.xlabel('iter')
plt.ylabel('percent')
plt.legend()
plt.savefig('./test2.jpg')
plt.show()