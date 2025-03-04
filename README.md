conda env setup:
$ conda create -n {env_name} python=3.12
$ conda activate {env name}
$ pip install -r requirements.txt

##############################  clustering run  ###############################

$ python3 clustering_run.py -i path/to/your/input_file.csv

#note: output file will saved in "output_cluster"

############################## data splitting #################################

$ python3 your_script_name.py -i path/to/your/input_file.csv

#note: output file will saved in "output_datasplitting"

docker setup:
$ cd /path/to/folder (where dockerfile is present)

#to build the image
$ docker build -t {image}:{tag} .

#to run 
$ docker run --rm -it -v {/local/path}:{/container_path} {image}:{tag} /bin/bash




