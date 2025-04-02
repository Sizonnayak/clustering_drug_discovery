##############################  conda env setup ###############################

$ conda create -n {env_name} python=3.12

$ conda activate {env name}

$ pip install -r requirements.txt

##############################  clustering run  ###############################

$ python3 clustering_run.py -i path/to/your/input_file.csv

#note: output file will saved in "output_cluster"

############################## data splitting #################################

$ python3 your_script_name.py -i path/to/your/input_file.csv

#note: output file will saved in "output_datasplitting"

############################## docker setup  ###################################

$ cd /path/to/folder (where dockerfile is present)

#to build the image

$ docker build -t {image}:{tag} .

#to run 

$ docker run --rm -it -v {/local/path}:{/container_path} {image}:{tag} /bin/bash

######################### additional docker commands ##########################
$ docker images

#to export the image

$ docker save {image}:{tag} > {file_name_as_per_you}.tar

#First copy it to the new host computer
#load the exported image to Docker 

$ docker load < {file_name_as_per_you}.tar

This project uses https://github.com/mqcomplab/bitbirch, which is licensed under the LGPL 3.0 License.
See the LICENSE file for the full terms.


