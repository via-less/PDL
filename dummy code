import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt

NUMBER_OF_SIMULATIONS = 1000

# calculated material flow matrix
L1 = [0, 10500, 0, 9000, 0, 0, 0]
L2 = [0, 0, 8500, 0, 0, 2000, 0]
L3 = [0, 0, 0, 5500, 0, 16000, 9000]
L4 = [0, 0, 0, 0, 14500, 0, 0]
L5 = [0, 0, 0, 0, 0, 9000, 5500]
L6 = [0, 0, 0, 0, 0, 0, 5000]
L7 = [0, 0, 0, 0, 0, 0, 0]
transport_matrix = np.array([L1, L2, L3, L4, L5, L6, L7])

FILENAME = 'simulation_results.xlsx'

# names of the machines: V, W, X, Y, Z
column_names = ['Raw_parts_x', 'Raw_parts_y', 'Machine_V_x', 'Machine_V_y',
                'Machine_W_x', 'Machine_W_y', 'Machine_X_x', 'Machine_X_y', 'Machine_Y_x', 'Machine_Y_y',
                'Machine_Z_x', 'Machine_Z_y', 'Finished_parts_x', 'Finished_parts_y', 'Distance_RP_V', 'Distance_RP_X',
                'Distance_V_W', 'Distance_V_Z', 'Distance_W_X', 'Distance_W_Z', 'Distance_W_FP', 'Distance_X_Y',
                'Distance_Y_Z', 'Distance_Y_FP', 'Distance_Z_FP', 'Total_transport_way[units]',
                'Total_transport_way[m]']

if os.path.exists(FILENAME):
    df = pd.read_excel(FILENAME)  # if file exists: use as basis
else:  #
    df = pd.DataFrame(columns=column_names)  # if dile doesn't exist: create new dataframe


class Machine:
    def __init__(self, size, x=None, y=None):
        self.size = size  # size of the hall

        if x is not None:  # x coordinate given (for raw & finished parts)
            self.x = x  # spawn it there
        else:  # x not given
            self.x = np.random.randint(0, size)  # spawn it random (0, 1, ..., size-1)
        if y is not None:  # same with y coordinate
            self.y = y
        else:
            self.y = np.random.randint(0, size)

    def __eq__(self, other):  # override == operator for positional comparison
        return self.x == other.x and self.y == other.y


class Layout:
    SIZE = 5  # 5x5 units
    Raw_parts_COLOR = 1  # keys to assign the correct color
    Finished_parts_COLOR = 2
    Machine_V_COLOR = 3
    Machine_W_COLOR = 4
    Machine_X_COLOR = 5
    Machine_Y_COLOR = 6
    Machine_Z_COLOR = 7
    # BGR (opencv)
    d = {1: (240, 32, 160),  # RP: purple
         2: (0, 0, 255),  # FP: red
         3: (255, 0, 0),  # V: blue
         4: (0, 255, 255),  # W: yellow
         5: (0, 255, 0),  # X: green
         6: (40, 90, 140),  # Y: brown
         7: (255, 255, 0)}  # Z: turquoise

    def __init__(self):
        self.Raw_Parts = Machine(size=self.SIZE, x=0, y=0)  # spawn raw parts at (0,0)
        self.Finished_Parts = Machine(size=self.SIZE, x=4, y=4)  # spawn finished parts at (4,4)

        # spawn all 5 machines at unique coordinates
        self.Machine_V = Machine(self.SIZE)
        while self.Machine_V == self.Raw_Parts or self.Machine_V == self.Finished_Parts:
            self.Machine_V = Machine(self.SIZE)  # respawn if necessary

        self.Machine_W = Machine(self.SIZE)
        while self.Machine_W == self.Raw_Parts or self.Machine_W == self.Finished_Parts or self.Machine_W == self.Machine_V:
            self.Machine_W = Machine(self.SIZE)

        self.Machine_X = Machine(self.SIZE)
        while self.Machine_X == self.Raw_Parts or self.Machine_X == self.Finished_Parts or self.Machine_X == self.Machine_V or self.Machine_X == self.Machine_W:
            self.Machine_X = Machine(self.SIZE)

        self.Machine_Y = Machine(self.SIZE)
        while self.Machine_Y == self.Raw_Parts or self.Machine_Y == self.Finished_Parts or self.Machine_Y == self.Machine_V or self.Machine_Y == self.Machine_W or self.Machine_Y == self.Machine_X:
            self.Machine_Y = Machine(self.SIZE)

        self.Machine_Z = Machine(self.SIZE)
        while self.Machine_Z == self.Raw_Parts or self.Machine_Z == self.Finished_Parts or self.Machine_Z == self.Machine_V or self.Machine_Z == self.Machine_W or self.Machine_Z == self.Machine_X or self.Machine_Z == self.Machine_Y:
            self.Machine_Z = Machine(self.SIZE)

    def create_img(self):  # method to show each layout during simulation shortly
        img = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # (width, height, color channels)
        img.fill(255)  # make zeros to 255 -> white
        # mark places where the machines & warehouses spawned with the correct colors
        img[self.Raw_Parts.x][self.Raw_Parts.y] = self.d[self.Raw_parts_COLOR]
        img[self.Finished_Parts.x][self.Finished_Parts.y] = self.d[self.Finished_parts_COLOR]
        img[self.Machine_V.x][self.Machine_V.y] = self.d[self.Machine_V_COLOR]
        img[self.Machine_W.x][self.Machine_W.y] = self.d[self.Machine_W_COLOR]
        img[self.Machine_X.x][self.Machine_X.y] = self.d[self.Machine_X_COLOR]
        img[self.Machine_Y.x][self.Machine_Y.y] = self.d[self.Machine_Y_COLOR]
        img[self.Machine_Z.x][self.Machine_Z.y] = self.d[self.Machine_Z_COLOR]
        img = Image.fromarray(img)   # create image from array
        img = img.resize((500, 500), resample=Image.BOX)  # make the image larger
        cv2.imshow("Layoutsimulation", np.array(img))
        cv2.waitKey(1)  # show layout (1ms)

    def formula_distance(self, x1, x2, y1, y2):  # d = ((x1-x2)²+(y1-y2)²)^0.5
        distance = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
        return distance  # distance between two points

    def calculate_transport_way(self):  # calculate all relevant distances
        distance_RP_V = self.formula_distance(self.Raw_Parts.x, self.Machine_V.x, self.Raw_Parts.y, self.Machine_V.y)
        distance_RP_X = self.formula_distance(self.Raw_Parts.x, self.Machine_X.x, self.Raw_Parts.y, self.Machine_X.y)
        distance_V_W = self.formula_distance(self.Machine_V.x, self.Machine_W.x, self.Machine_V.y, self.Machine_W.y)
        distance_V_Z = self.formula_distance(self.Machine_V.x, self.Machine_Z.x, self.Machine_V.y, self.Machine_Z.y)
        distance_W_X = self.formula_distance(self.Machine_W.x, self.Machine_X.x, self.Machine_W.y, self.Machine_X.y)
        distance_W_Z = self.formula_distance(self.Machine_W.x, self.Machine_Z.x, self.Machine_W.y, self.Machine_Z.y)
        distance_W_FP = self.formula_distance(self.Machine_W.x, self.Finished_Parts.x, self.Machine_W.y, self.Finished_Parts.y)
        distance_X_Y = self.formula_distance(self.Machine_X.x, self.Machine_Y.x, self.Machine_X.y, self.Machine_Y.y)
        distance_Y_Z = self.formula_distance(self.Machine_Y.x, self.Machine_Z.x, self.Machine_Y.y, self.Machine_Z.y)
        distance_Y_FP = self.formula_distance(self.Machine_Y.x, self.Finished_Parts.x, self.Machine_Y.y, self.Finished_Parts.y)
        distance_Z_FP = self.formula_distance(self.Machine_Z.x, self.Finished_Parts.x, self.Machine_Z.y, self.Finished_Parts.y)
        # create a matrix with the distances in the right spots
        matrix_distance = np.array([[0, distance_RP_V, 0, distance_RP_X, 0, 0, 0],
                                    [0, 0, distance_V_W, 0, 0, distance_V_Z, 0],
                                    [0, 0, 0, distance_W_X, 0, distance_W_Z, distance_W_FP],
                                    [0, 0, 0, 0, distance_X_Y, 0, 0],
                                    [0, 0, 0, 0, 0, distance_Y_Z, distance_Y_FP],
                                    [0, 0, 0, 0, 0, 0, distance_Z_FP],
                                    [0, 0, 0, 0, 0, 0, 0]])
        # multiply the matrices to get the total transport way between each relevant machine & warehouse
        matrix_transport_way = matrix_distance * transport_matrix
        transport_route = np.sum(matrix_transport_way)  # total transport way in the next period based on the random layout
        return distance_RP_V, distance_RP_X, distance_V_W, distance_V_Z, distance_W_X, distance_W_Z, distance_W_FP, distance_X_Y, distance_Y_Z, distance_Y_FP, distance_Z_FP, transport_route


list_RP_x = []  # lists for the results
list_RP_y = []
list_V_x = []
list_V_y = []
list_W_x = []
list_W_y = []
list_X_x = []
list_X_y = []
list_Y_x = []
list_Y_y = []
list_Z_x = []
list_Z_y = []
list_FP_x = []
list_FP_y = []
list_distance_RP_V = []
list_distance_RP_X = []
list_distance_V_W = []
list_distance_V_Z = []
list_distance_W_X = []
list_distance_W_Z = []
list_distance_W_FP = []
list_distance_X_Y = []
list_distance_Y_Z = []
list_distance_Y_FP = []
list_distance_Z_FP = []
list_transport_way_result = []
list_transport_way_result_meter = []

for simulation in range(NUMBER_OF_SIMULATIONS):  # run number of desired simulations
    layout = Layout()  # spawn random layout
    layout.create_img()  # show it
    # calculate distances
    distance_RP_V, distance_RP_X, distance_V_W, distance_V_Z, distance_W_X, distance_W_Z, distance_W_FP, distance_X_Y, distance_Y_Z, distance_Y_FP, distance_Z_FP, transport_route = layout.calculate_transport_way()
    list_RP_x.append(layout.Raw_Parts.x)  # append to lists
    list_RP_y.append(layout.Raw_Parts.y)
    list_V_x.append(layout.Machine_V.x)
    list_V_y.append(layout.Machine_V.y)
    list_W_x.append(layout.Machine_W.x)
    list_W_y.append(layout.Machine_W.y)
    list_X_x.append(layout.Machine_X.x)
    list_X_y.append(layout.Machine_X.y)
    list_Y_x.append(layout.Machine_Y.x)
    list_Y_y.append(layout.Machine_Y.y)
    list_Z_x.append(layout.Machine_Z.x)
    list_Z_y.append(layout.Machine_Z.y)
    list_FP_x.append(layout.Finished_Parts.y)
    list_FP_y.append(layout.Finished_Parts.y)
    list_distance_RP_V.append(distance_RP_V)
    list_distance_RP_X.append(distance_RP_X)
    list_distance_V_W.append(distance_V_W)
    list_distance_V_Z.append(distance_V_Z)
    list_distance_W_X.append(distance_W_X)
    list_distance_W_Z.append(distance_W_Z)
    list_distance_W_FP.append(distance_W_FP)
    list_distance_X_Y.append(distance_X_Y)
    list_distance_Y_Z.append(distance_Y_Z)
    list_distance_Y_FP.append(distance_Y_FP)
    list_distance_Z_FP.append(distance_Z_FP)
    list_transport_way_result.append(transport_route)
    list_transport_way_result_meter.append(transport_route * 3)


new_df = pd.DataFrame(columns=column_names)  # make new df from simulation results
new_df['Raw_parts_x'] = list_RP_x  # assign lists to correct column
new_df['Raw_parts_y'] = list_RP_y
new_df['Machine_V_x'] = list_V_x
new_df['Machine_V_y'] = list_V_y
new_df['Machine_W_x'] = list_W_x
new_df['Machine_W_y'] = list_W_y
new_df['Machine_X_x'] = list_X_x
new_df['Machine_X_y'] = list_X_y
new_df['Machine_Y_x'] = list_Y_x
new_df['Machine_Y_y'] = list_Y_y
new_df['Machine_Z_x'] = list_Z_x
new_df['Machine_Z_y'] = list_Z_y
new_df['Finished_parts_x'] = list_FP_x
new_df['Finished_parts_y'] = list_FP_y
new_df['Distance_RP_V'] = list_distance_RP_V
new_df['Distance_RP_X'] = list_distance_RP_X
new_df['Distance_V_W'] = list_distance_V_W
new_df['Distance_V_Z'] = list_distance_V_Z
new_df['Distance_W_X'] = list_distance_W_X
new_df['Distance_W_Z'] = list_distance_W_Z
new_df['Distance_W_FP'] = list_distance_W_FP
new_df['Distance_X_Y'] = list_distance_X_Y
new_df['Distance_Y_Z'] = list_distance_Y_Z
new_df['Distance_Y_FP'] = list_distance_Y_FP
new_df['Distance_Z_FP'] = list_distance_Z_FP
new_df['Total_transport_way[units]'] = list_transport_way_result
new_df['Total_transport_way[m]'] = list_transport_way_result_meter

df = df.append(new_df, ignore_index=True)  # append new dataframe to old
df.to_excel(FILENAME)  # save as excel (override old if existing)

# print best result and row in Excel file
print('Minimal transportation way from all simulations: ', df['Total_transport_way[m]'].min(),
      ' m.\nRow in the Excel file (simulation_results.xlsx): ', df['Total_transport_way[m]'].idxmin() + 2)


def show_best_layout(df):  # show the final layout
    index = df['Total_transport_way[m]'].idxmin()  # index of the best result
    img_best_layout = np.zeros((Layout.SIZE, Layout.SIZE, 3), dtype=np.uint8)
    img_best_layout.fill(255)
    img_best_layout[df.at[index, 'Raw_parts_x']][df.at[index, 'Raw_parts_y']] = Layout.d[Layout.Raw_parts_COLOR]
    img_best_layout[df.at[index, 'Finished_parts_x']][df.at[index, 'Finished_parts_y']] = Layout.d[Layout.Finished_parts_COLOR]
    img_best_layout[df.at[index, 'Machine_V_x']][df.at[index, 'Machine_V_y']] = Layout.d[Layout.Machine_V_COLOR]
    img_best_layout[df.at[index, 'Machine_W_x']][df.at[index, 'Machine_W_y']] = Layout.d[Layout.Machine_W_COLOR]
    img_best_layout[df.at[index, 'Machine_X_x']][df.at[index, 'Machine_X_y']] = Layout.d[Layout.Machine_X_COLOR]
    img_best_layout[df.at[index, 'Machine_Y_x']][df.at[index, 'Machine_Y_y']] = Layout.d[Layout.Machine_Y_COLOR]
    img_best_layout[df.at[index, 'Machine_Z_x']][df.at[index, 'Machine_Z_y']] = Layout.d[Layout.Machine_Z_COLOR]
    img_best_layout = Image.fromarray(img_best_layout)
    img_best_layout = img_best_layout.resize((5, 5), resample=Image.BOX)
    img_best_layout = np.array(img_best_layout)
    img_best_layout = cv2.cvtColor(img_best_layout, cv2.COLOR_BGR2RGB)  # BGR in RGB (for matplotlib)
    plt.imshow(img_best_layout)
    # create a plot for each machine / warehouse to use as a legend
    plt.plot(0, 0, color=(160/255, 32/255, 240/255), label='Raw Parts Warehouse')
    plt.plot(0, 0, color=(255/255, 0, 0), label='Finished Parts Warehouse')
    plt.plot(0, 0, color=(0, 0, 255/255), label='Machine_V')
    plt.plot(0, 0, color=(255/255, 255/255, 0), label='Machine_W')
    plt.plot(0, 0, color=(0, 255/255, 0), label='Machine_X')
    plt.plot(0, 0, color=(140/255, 90/255, 40/255), label='Machine_Y')
    plt.plot(0, 0, color=(0, 255/255, 255/255), label='Machine_Z')
    plt.legend(bbox_to_anchor=(1.33, 1.0))
    plt.show()


show_best_layout(df=df)  # show best layout
