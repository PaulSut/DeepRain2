import wradlib as wrl
import numpy as np
import tarfile
import os
import pickle
import matplotlib.pyplot as plt



if __name__ == '__main__':
    abs_path = os.path.abspath('.')
    path_of_radar_zip = abs_path + "/Data/RadDataZIP"
    list_of_zips = os.listdir(path_of_radar_zip)

    '''
     tar_year = tarfile.open(path_of_radar_zip +"/YW2017.002_201606.tar", "r:")
    tar_year.extractall(abs_path +"/Data/RadDataZipMonth")
    list_month= os.listdir(abs_path +"/Data/RadDataZipMonth")
    for month in list_month:
        tar_month = tarfile.open(abs_path +"/Data/RadDataZipMonth/" + month, "r:gz")
        tar_month.extractall(abs_path +"/Data/RadData")
    
    
    list_of_rad_pics = os.listdir(abs_path +"/Data/RadData")
    list_of_rad_pics.sort()
    all_data = []
    i = 1
    for rad_pic in list_of_rad_pics:
        data_pic = wrl.io.read_radolan_composite(abs_path +"/Data/RadData/" + rad_pic)
        unique_val, counts = np.unique(data_pic[0], return_counts=True)
        zip_unique_counts = [[unique_val[index], counts[index]] for index in range(len(unique_val))]
        zip_unique_counts = zip_unique_counts
        all_data_values = [value[0] for value in all_data]

        for index in range(len(list(zip_unique_counts))):
            try:
                all_data_index = all_data_values.index(zip_unique_counts[index][0])
            except:
                all_data.append(zip_unique_counts[index])
            else:
                list(all_data)[all_data_index][1] = all_data[all_data_index][1] + zip_unique_counts[index][1]
        if i % 100 == 0:
            i = 1
            print("Done")
        i+=1



    print(all_data)
    pickle.dump(all_data, open("YW2017.002_201606_countUniqueValues.p", "wb"))
    '''

    uniqueValues = pickle.load(open("YW2017.002_201606_countUniqueValues.p", "rb"))
    uniqueValues = [value for value in uniqueValues if value[0] > 0]
    uniqueValues = np.asanyarray(uniqueValues)
    print(uniqueValues[:,1])
    plt.plot(uniqueValues[:,0], uniqueValues[:,1], 'ro')
    plt.ylabel('Anzahl')
    plt.show()

    val = 0
    number_of_values = 0
    for value in uniqueValues:
        val = val + value[1]*value[0]
        number_of_values = number_of_values + value[1]

    print(val)
    print(number_of_values)
    mean = val/number_of_values
    print("MEAN: ", mean)








