import os
from os.path import join
from ipdb import set_trace

root_folder = join("/", "home", "torbjone", "work",
              "spike_sorting", "neuron_models")
model_folders = ['MediumSimple', 'SmallSimple',
                 'SmallComplex', 'MediumComplex', 'Large']

folders = [join(root_folder, x) for x in model_folders]
file_list = []
for folder in folders:
    for model in os.listdir(folder):
        temp = join(folder, model)
        file_ = os.listdir(temp)
        file_ = [x for x in file_ if x[-4:] != '.asc']
        file_ = [x for x in file_ if not 'Morph' in x]
        file_ = [x for x in file_ if x[-4:] == '.hoc']
        filename = join(temp, file_[0])
        input_file = file(filename, 'r')
        lines = input_file.readlines()
        output_file = file(filename, 'w')
        for line in lines:
            if 'axon {L=5340}' in line:
                #print line
                line = 'axon {L=100}'
            #if 'objectvar clamp' in line:
            #    line = '//' + line
            #if 'soma clamp = new IClamp(0.5)' in line:
            #    line = '//' + line
            #if '//load_proc("nrnmainmenu")' in line:
            #    line = line[2:]
            #nums = line.split(" ") 
            #retain = nums[:5] 
            #new_line = ", ".join(retain)+"\n" 
            output_file.write(line)
        output_file.close()
        
        #set_trace()
        #print os.listdir(temp)
        #file_list.append(join(model, file))
#print file_list, '\n'
