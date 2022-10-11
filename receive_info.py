import queue
import socket
from typing_extensions import IntVar 
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as Tk
from threading import Thread
import queue
import time
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestRegressor

loaded_rf = joblib.load("./random_forest.joblib")
max_time = 60000000

test = np.random.randint(200, 400, size=(500))
test2 =np.random.randint(50, 80, size=(500))
test3 =np.random.randint(4, 6, size=(500))
test4 =np.random.randint(3, 7, size=(500))

change_request = True
cpu_freq = 0
work_mode = 1
cpu_utilization = 0

work_task = 0
main_task = 0
idle1_task = 0
idle0_task = 0

now = 0
later = 0

def create_work_msg():
    global change_request, cpu_freq, work_mode, cpu_utilization
    if(change_request):
        change_request = False
        return "y," + str(cpu_freq) + "," + str(work_mode) + "," + str(cpu_utilization)
    else:
        return "n"


def get_msg(work ,frequency, target_util):
    return "Application:" + work.split(".")[0] + " | Frequency: " + frequency + " | Target CPU Util: " + target_util  

HOST = "192.168.178.23"
PORT =  8090
receive_msg = "Receive Data"

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

write_index = 0

#there is a smarter way to handle these queues but for now it will do
def retrieve_data(q_power, q_current, q_voltage, q_shunt, q_work, q_main, q_idle, q_idle2, q_info):
    global cpu_freq, work_mode, cpu_utilization
    #print("Thread started", flush=True)
    # msg = "c,add_test.csv,160,75,149990917,10,1462,0,49999512,3,199907179,14,160,160"
    # i = 0
    msg = create_work_msg()
    s.send(msg.encode())
    while True:
        # if i%10 == 0:
        #     data = "x"
        # else:
        #     data = "c"
        data = str(s.recv(1024).decode()).split(";")[0]
        if data[0] != 'c':
            #time.sleep(2)
            #data = str(test3[i]) + "," + str(test2[i]) + "," + str(test[i]) + "," + str(test4[i]) 
            q_power.put(float(data.split(",")[2]))
            q_current.put(float(data.split(",")[1]))
            q_voltage.put(float(data.split(",")[0]))
            q_shunt.put(float(data.split(",")[3]))
        else:
            #data = msg
            print(data.split(","))
            q_info.put(data.split(",")[1] + "," + data.split(",")[2] + "," + data.split(",")[3])
            q_work.put(data.split(",")[4])
            q_main.put(data.split(",")[6])
            q_idle.put(data.split(",")[8])
            q_idle2.put(data.split(",")[10])
            msg = create_work_msg()
            s.send(msg.encode())
            
        # i += 1
        # if i == 499:
        #     i = 0

q_power = queue.Queue()
q_current = queue.Queue()
q_voltage = queue.Queue()
q_shunt = queue.Queue()
q_work = queue.Queue()
q_main = queue.Queue()
q_idle = queue.Queue()
q_idle2 = queue.Queue()
q_info = queue.Queue()
 
t = Thread(target= retrieve_data, args = (q_power, q_current, q_voltage, q_shunt, q_work, q_main, q_idle, q_idle2, q_info, ))
t.start()

# First set up the figure, the axis, and the plot element we want to animate
y_limits = [600, 100, 5.1, 9, 2000]
y_limits_min = [0, 0, 4.8, 2, 0]
titles = ["Power", "Current", "Voltage", "Shunt", "Energy"]
y_labels = ["Milliwatt", "Milliampere", "Volt", "Volt", "Millijoule"]
line_labels = "Measured"
avg_line_labels = "Average"
rf_predict = "RF Predict"
pol_predict = "Poly Predict"
colors = ["red", "blue", "green", "yellow", "purple"]
plots = [231,232,233,234,235,236]
fig = plt.figure(figsize=(10,8))
axes = []
pos = 121
for i in range(0, len(y_limits)):
    axes.append(fig.add_subplot(plots[i]))
    axes[i].set_xlim([0, 1])
    axes[i].set_ylim([y_limits_min[i], y_limits[i]])
    axes[i].set_title(titles[i])
    axes[i].set_xlabel("Time")
    axes[i].set_ylabel(y_labels[i])

bar = fig.add_subplot(plots[len(y_limits)])
bar.set_ylim(0,100)
rectangles = plt.bar([0.5,1,1.5,2],[0,0,0,0],width=0.1)
bar.set_ylabel("Percentage Running")
bar.set_xticks([0.5,1,1.5,2], labels=['Run1', 'Run0', 'Idle1', 'Idle0'])

lines = [axes[i].plot([], [], lw=2, color=colors[i], label=line_labels)[0] for i in range(0, len(y_limits))]
predictions = [axes[4].plot([], [], lw=2, color="black", label=rf_predict)[0]] + [axes[4].plot([], [], lw=1, color="red", label=pol_predict)[0]]
avg_lines = [axes[i].plot([], [], lw=2, color="black", label=avg_line_labels)[0] for i in range(0, len(y_limits)-1)]

for ax in axes:
    ax.legend(loc="upper left")

patches = lines + avg_lines + list(rectangles) + predictions

fig.tight_layout(pad=2.0)

# initialization function: plot the background of each frame
def init():
    for line in lines:
        line.set_data([], [])

    for prediction in predictions:
        prediction.set_data([], [])

    for line in avg_lines:
        line.set_data([], [])
    
    for rectangle in rectangles:
        rectangle.set_height(0)

    return patches

j = 0
x = []
y = [[] for _ in range(0, len(y_limits))]
rf_values = []
pol_values = []
min_x = True

def write_back(freq,util,work):
    global write_index, y
    file_name = "./stats/stats.csv"
    with open(file_name, 'a') as f:
        for i in range(write_index, len(y[0])):
            f.write(str(y[0][i]) + "," + str(y[1][i]) + "," + str(y[2][i]) + "," + str(y[3][i]) + "," + str(freq) + "," + str(util) + "," + str(work) + "\n")
    write_index = len(y[0])

def set_x_lim():
    global min_x
    if min_x:
        min_x = False
    else:
        min_x = True

def get_x_lim():
    global min_x, x
    if min_x:
        return 0
    else:
        return max(x)-10


def enumerate_work(work):
    if(work == "noop_test.csv"):
        return 0
    if(work == "add_test.csv"):
        return 1
    if(work == "sub_test.csv"):
        return 2
    if(work == "mul_test.csv"):
        return 3
    if(work == "div_test.csv"):
        return 4
    if(work == "addf_test.csv"):
        return 5
    if(work == "subf_test.csv"):
        return 6
    if(work == "mulf_test.csv"):
        return 7
    if(work == "divf_test.csv"):
        return 8
    if(work == "linkedlist_test.csv"):
        return 9

def renumerate_work(work):
    if(work == 0):
        return "noop_test.csv"
    if(work == 1):
        return "add_test.csv"
    if(work == 2):
        return "sub_test.csv"
    if(work == 3):
        return "mul_test.csv"
    if(work == 4):
        return "div_test.csv"
    if(work == 5):
        return "addf_test.csv"
    if(work == 6):
        return "subf_test.csv"
    if(work == 7):
        return "mulf_test.csv"
    if(work == 8):
        return "divf_test.csv"
    if(work == 9):
        return "linkedlist_test.csv"

def get_freq(index):
    return [160,240][index]

def get_util(index):
    return [25,50,75,100][index]


actual_work = work_mode
actual_freq = get_freq(cpu_freq)
actual_util = get_util(cpu_utilization)
rf_error = [0]
pol_error = 0

def calculate_rmse(predicted, measured):
    return np.sqrt((measured - predicted)**2)

def update_status():
    global actual_work, actual_freq, actual_util,status,root,rf_values,y,pol_values,label_rf,label_pol
    if not q_info.empty():
        tmp = str(q_info.get())
        actual_work = enumerate_work(tmp.split(",")[0])
        actual_freq = int(tmp.split(",")[1])
        actual_util = int(tmp.split(",")[2])

        status["text"] = get_msg(tmp.split(",")[0], tmp.split(",")[1], tmp.split(",")[2])
        write_back(tmp.split(",")[1], tmp.split(",")[2], tmp.split(",")[0])
    
    label_rf["text"]  = "Random Forrest RMSE: " + str(rf_error[0])
    label_pol["text"]  = "Polynomial Model RMSE: " + str(pol_error)

    # After 10 second, update the status
    root.after(10000, update_status)

def sel_freq():
    global cpu_freq, change_request
    cpu_freq = int(freq_var.get())
    change_request = True

def sel_util():
    global cpu_utilization, change_request
    cpu_utilization = int(util_var.get())
    change_request = True

def sel_work():
    global work_mode, change_request
    work_mode = int(work_var.get())
    change_request = True

avg_values = [0,0,0,0]
rf_avg = 0
pol_avg = 0

coefs160 = [ 5.14505510e-03, -1.99111601e+00,  4.08712398e+02]
coefs240 = [-1.62309970e-04,  1.47397293e-01, -4.35805361e+01,  4.50433257e+03]

def predict_polynomial(freq, util):
    if freq == 240:
        return (coefs240[0] * (freq+util)**3) + (coefs240[1] * (freq+util)**2) + (coefs240[2] * (freq+util)) + coefs240[3]
    else:
        return (coefs160[0] * (freq+util)**2) + (coefs160[1] * (freq+util)) + coefs160[2]

# animation function.  This is called sequentially
def animate(i):
    global j,x,y, work_task, main_task, idle1_task, idle0_task, avg_values, rf_avg, predictions, pol_avg, pol_values, rf_error, pol_error
    while not q_power.empty():
        tmp = float(q_power.get())
        y[0].append(tmp)
        avg_values[0] += tmp
        time = (datetime.now() - now).total_seconds()
        y[4].append((avg_values[0]/len(y[0]))*time)
        #print("predict" + str(actual_freq) + "," + str(actual_util) + "," + str(actual_work))
        rf_avg += loaded_rf.predict([[actual_freq,actual_util,actual_work]])
        rf_values.append((rf_avg/len(y[0]))*time)
        pol_avg += predict_polynomial(actual_freq, actual_util)
        pol_values.append((pol_avg/len(y[0]))*time)
        rf_error = calculate_rmse(y[4][-1], rf_values[-1])
        pol_error = calculate_rmse(y[4][-1], pol_values[-1])
        axes[4].set_ylim(0, max([max(y[4]),max(rf_values),max(pol_values)]))
        x.append(j)
        j += 1
        if((j)%10 == 0):
            break

    if(len(x) != 0):
        for ax in axes:
            #print(str(min(x)) + "," + str(max(x)))
            ax.set_xlim(get_x_lim(), max(max(x),1))

    while len(y[1]) < len(y[0]):
        tmp = float(q_current.get())
        avg_values[1] += tmp
        y[1].append(tmp)

    while len(y[2]) < len(y[0]):
        tmp = float(q_voltage.get())
        avg_values[2] += tmp
        y[2].append(tmp)

    while len(y[3]) < len(y[0]):
        tmp = float(q_shunt.get())
        avg_values[3] += tmp
        y[3].append(tmp)

    for iter in range(0, len(y_limits)):
        lines[iter].set_data(x, y[iter])
    predictions[0].set_data(x, rf_values)
    predictions[1].set_data(x, pol_values)

    for iter in range(0, len(y_limits)-1):
        if(avg_values[0] == 0):
            break
        avg_lines[iter].set_data(x, avg_values[iter]/len(y[0]))

    while not q_idle2.empty():
        work_task = (int(q_work.get())/max_time)*100
        main_task = (int(q_main.get())/max_time)*100
        idle1_task = (int(q_idle.get())/max_time)*100
        idle0_task = (int(q_idle2.get())/max_time)*100
    
    rectangles[0].set_height(work_task)
    rectangles[1].set_height(main_task)
    rectangles[2].set_height(idle1_task)
    rectangles[3].set_height(idle0_task)

    return patches

root = Tk.Tk()
root.configure(bg='white') 

label = Tk.Label(root,text="Energy Consumption Framework").grid(column=0, row=0)
label_rf = Tk.Label(root,text="Random Forrest RMSE: 0")
label_rf.grid(column=1, row=31)
label_pol = Tk.Label(root,text="Polynomial Model RMSE: 0")
label_pol.grid(column=2, row=31)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(column=0,row=1, columnspan=3, rowspan=30)

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20)

button = Tk.Button(root, text ="Switch View", command = set_x_lim)
button.grid(column=0, row=31)

status = Tk.Label(root, text=get_msg(renumerate_work(actual_work), str(actual_freq), str(actual_util)))
status.grid(column=1, row=0)

root.after(1, update_status)

#return "y," + str(cpu_freq) + "," + str(work_mode) + "," + str(cpu_utilization)

freq_var = Tk.IntVar()
R1 = Tk.Radiobutton(root, text="CPU Freq 160", variable=freq_var, value=0,
                  command=sel_freq)
R2 = Tk.Radiobutton(root, text="CPU Freq 240", variable=freq_var, value=1,
                  command=sel_freq)

util_var = Tk.IntVar()
R3 = Tk.Radiobutton(root, text="CPU UTIL 25", variable=util_var, value=0,
                  command=sel_util)
R4 = Tk.Radiobutton(root, text="CPU UTIL 50", variable=util_var, value=1,
                  command=sel_util)
R5 = Tk.Radiobutton(root, text="CPU UTIL 75", variable=util_var, value=2,
                  command=sel_util)
R6 = Tk.Radiobutton(root, text="CPU UTIL 100", variable=util_var, value=3,
                  command=sel_util)

work_var = Tk.IntVar()
R7 = Tk.Radiobutton(root, text="NOOP Test", variable=work_var, value=0,
                  command=sel_work)
R8 = Tk.Radiobutton(root, text="Add Int Test", variable=work_var, value=1,
                  command=sel_work)
R9 = Tk.Radiobutton(root, text="Sub Int Test", variable=work_var, value=2,
                  command=sel_work)
R10 = Tk.Radiobutton(root, text="Mul Int Test", variable=work_var, value=3,
                  command=sel_work)
R11 = Tk.Radiobutton(root, text="Div Int Test", variable=work_var, value=4,
                  command=sel_work)
R12 = Tk.Radiobutton(root, text="Add Float Test", variable=work_var, value=5,
                  command=sel_work)
R13 = Tk.Radiobutton(root, text="Sub Float Test", variable=work_var, value=6,
                  command=sel_work)
R14 = Tk.Radiobutton(root, text="Mul Float Test", variable=work_var, value=7,
                  command=sel_work)
R15 = Tk.Radiobutton(root, text="Div Float Test", variable=work_var, value=8,
                  command=sel_work)
R16 = Tk.Radiobutton(root, text="LinkedList Test", variable=work_var, value=9,
                  command=sel_work)


R1.grid(column=3, row=0)
R2.grid(column=3, row=1)
R3.grid(column=3, row=3)
R4.grid(column=3, row=4)
R5.grid(column=3, row=5)
R6.grid(column=3, row=6)
R7.grid(column=3, row=8)
R8.grid(column=3, row=9)
R9.grid(column=3, row=10)
R10.grid(column=3, row=11)
R11.grid(column=3, row=12)
R12.grid(column=3, row=13)
R13.grid(column=3, row=14)
R14.grid(column=3, row=15)
R15.grid(column=3, row=16)
R16.grid(column=3, row=17)

now = datetime.now()

Tk.mainloop()