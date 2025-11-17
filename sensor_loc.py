import numpy as np
import torch





def cylinder_4BC_sensors():
    
    
    coords = []
    for count, i in enumerate( range(0, 112, 28) ):
        if count==0:
            continue
        coords.append([5,  i])
        coords.append([187,i])

    coords = np.array( coords )

    coords = np.flip( coords, axis=1)
    
    
    return coords[[0,1,4,5],0], coords[[0,1,4,5],1]


def cylinder_8_sensors(data, n_sensors, rnd_seed):
    
    np.random.seed(rnd_seed)
    
    im = np.copy(data[0,]).squeeze()
    print(im.shape)
    
    print('Picking up sensor locations \n')
    coords = []

    # DENSE_ENCODER = False
    DENSE_ENCODER = True
    if DENSE_ENCODER:
        coords = [
            [62, 45],
            [62, 2],
            [28, 34],
            [38, 17],
            [19, 42],
            [59, 57],
            [22, 33],
            [32, 49],
            [62, 47],
            [9, 32],
            [46, 32],
            [47, 25],
            [19, 14],
            [61, 36],
            [32, 16],
            [4, 49],
            [55, 3],
            [2, 20],
            [39, 2],
            [20, 47]
        ]
    
    for n in range(n_sensors):
        if DENSE_ENCODER:
            n = n + 20
        while True:
            new_x = np.random.randint(0,data.shape[1],1)[0]
            new_y = np.random.randint(0,data.shape[2],1)[0]
            if im[new_x,new_y] != 0:
                coords.append([new_x,new_y])
                im[new_x,new_y] = 0
                break
    coords = np.array(coords)  
    return coords[:,0], coords[:,1]

def shallow_water_n_sensors(data, n_sensors, rnd_seed):
    
    np.random.seed(rnd_seed)
    
    im = np.copy(data[0,]).squeeze()
    print(im.shape)
    
    print('Picking up sensor locations \n')
    coords = []
    
    for n in range(n_sensors):
        while True:
            new_x = np.random.randint(0,data.shape[1],1)[0]
            new_y = np.random.randint(0,data.shape[2],1)[0]
            
            # 使用 np.any() 来检查是否至少有一个通道的值不为0
            if np.any(im[new_x, new_y] != 0): # 更简洁的方式是 np.any(im[new_x, new_y])
                coords.append([new_x,new_y])
                # 将这个位置清零，防止重复选择
                im[new_x,new_y] = 0 # Numpy的广播机制会自动将0扩展到所有通道
                break
    coords = np.array(coords)  
    return coords[:,0], coords[:,1]

def cylinder_16_sensors():
    
    coords = np.array( [ [76,71],  [175,69], [138,49],                   
                         [41, 56], [141,61], [30,41],  
                         [177,40], [80,55],  [60,41], [70,60],
                         [100,60], [120,51], [160,80],[165,50],
                         [180,60], [30,70] ] )
    
    coords = np.flip( coords, axis=1)
    
    return coords[:,0], coords[:,1]






def sea_n_sensors(data, n_sensors, rnd_seed):
    
    np.random.seed(rnd_seed)
    
    im = np.copy(data[0,]).squeeze()

    print(im.shape)
    
    print('Picking up sensor locations \n')
    coords = []
    
    for n in range(n_sensors):
        while True:
            new_x = np.random.randint(0,data.shape[1],1)[0]
            new_y = np.random.randint(0,data.shape[2],1)[0]
            if im[new_x,new_y] != 0:
                coords.append([new_x,new_y])
                im[new_x,new_y] = 0
                break
    coords = np.array(coords)  
    return coords[:,0], coords[:,1]
                


def sensors_3D(data, n_sensors, rnd_seed):
    
    np.random.seed(rnd_seed)
    
    im = np.copy(data[0,]).squeeze()
    #print(im.shape, im)
    
    print('Picking up sensor locations \n')
    coords = []
    
    # for n in range(n_sensors):
    #     while True:
    #         new_x = np.random.randint(0,data.shape[1],1)[0]
    #         new_y = np.random.randint(0,data.shape[2],1)[0]
    #         new_z = np.random.randint(0,data.shape[3],1)[0]
    #         if im[new_x,new_y,new_z] != 0:
    #             coords.append([new_x,new_y,new_z])
    #             im[new_x,new_y,new_z] = 0
    #             break
    #         # if [new_x, new_y, new_z] not in coords:
    #         #     coords.append([new_x, new_y, new_z])
    #         #     break
    for n in range(n_sensors):
        while True:
            # 限制在空间中心区域（各维度中间50%范围）
            start_x = data.shape[1] // 4
            end_x = data.shape[1] * 3 // 4
            new_x = np.random.randint(start_x, end_x, 1)[0]
            
            start_y = data.shape[2] // 4
            end_y = data.shape[2] * 3 // 4
            new_y = np.random.randint(start_y, end_y, 1)[0]
            
            start_z = data.shape[3] // 4
            end_z = data.shape[3] * 3 // 4 
            new_z = np.random.randint(start_z, end_z, 1)[0]
            if im[new_x,new_y,new_z] != 0:
                coords.append([new_x,new_y,new_z])
                im[new_x,new_y,new_z] = 0
                break
    coords = np.array(coords)  
    return coords[:,0], coords[:,1], coords[:,2]
                

def sensors_3D_custom(data, n_sensors, rnd_seed):
    
    np.random.seed(rnd_seed)
    
    im = np.copy(data[50,]).squeeze()
    #print(im.shape, im)
    
    print('Picking up sensor locations \n')
    coords = []
    
    # for n in range(n_sensors):
    #     while True:
    #         new_x = np.random.randint(0,data.shape[1],1)[0]
    #         new_y = np.random.randint(0,data.shape[2],1)[0]
    #         new_z = np.random.randint(0,data.shape[3],1)[0]
    #         if im[new_x,new_y,new_z] != 0:
    #             coords.append([new_x,new_y,new_z])
    #             im[new_x,new_y,new_z] = 0
    #             break
    #         # if [new_x, new_y, new_z] not in coords:
    #         #     coords.append([new_x, new_y, new_z])
    #         #     break
    for n in range(n_sensors//2):
        y = data.shape[1] // 2
        dis = data.shape[3] // 5
        x = 1 + dis*(n)
        z = data.shape[2] // 2
        coords.append([z,y,x])
    for n in range(n_sensors//2):
        y = data.shape[1] // 2
        dis = data.shape[3] // 5
        x = 1 + dis*(n)
        z = data.shape[2] - 1
        coords.append([z,y,x])
    print(coords)
        
    coords = np.array(coords)  
    return coords[:,0], coords[:,1], coords[:,2]
        
    
    

        
        
    
    
    
    
    
    