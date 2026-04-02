import os, shutil, sys

count =  0
with open("DATASET/filtered_data_split.csv")  as f:
    lines = f.readlines()

    for line in lines:
        line = line.replace("\n", "").split(",")

        if line[-1] == 'test':
            shutil.copy("DATASET/"+line[1][1:], f"DATASET/Testing_Data/{ line[1].split('/')[-1] }")
            
            count+=1

print(count, len(os.listdir("DATASET/Testing_Data/")))