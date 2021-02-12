#!/bin/bash

# arrowhyx@foxmail.com
# do not forget to replace those path with correct path
strorage_dir=$1

aishell_1=$strorage_dir/source_datasets/aishell_1/
aishell_3=$strorage_dir/source_datasets/aishell_3/
vctk=$strorage_dir/source_datasets/vctk/
librispeech=$strorage_dir/source_datasets/librispeech_360/
musan=$strorage_dir/noise_datasets/musan/
audioset=$strorage_dir/noise_datasets/audioset/
linear=$strorage_dir/linear/
circle=$strorage_dir/circle/
non_uniform=$strorage_dir/non_uniform/

selected_lists_path='../selected_lists/'

if [ ! -d data ] ; then
    mkdir data
fi

## train 
#speech
> ./data/train_clean.lst
for name_path in ${aishell_3} ${aishell_1} ${vctk} ${librispeech} ; do
    name=`basename ${name_path}`
    find ${name_path} -regex ".*\.wav\|.*\.flac" >/tmp/${name} 
    echo $name
    #grep -r -f ./selected_lists/train/${name}.name /tmp/${name} >> ./data/train_clean.lst
    python ./quick_select.py ${selected_lists_path}/train/${name}.name /tmp/${name} >> ./data/train_clean.lst
done
##noise
> ./data/train_noise.lst
for name_path in  ${musan} ${audioset} ; do
    name=`basename ${name_path}`
    echo $name
    find ${name_path} -iname "*.wav" >/tmp/${name} 
    #grep -r -f ./selected_lists/train/${name}.name /tmp/${name} >> ./data/train_noise.lst
    python ./quick_select.py ${selected_lists_path}/train/${name}.name /tmp/${name} >> ./data/train_noise.lst
done

grep -r -f ${selected_lists_path}/dev/noise.name /tmp/musan > ./data/dev_noise.lst
grep -r -f ${selected_lists_path}/dev/clean.name /tmp/aishell_1 > ./data/dev_clean.lst
grep -r -f ${selected_lists_path}/dev/clean.name /tmp/vctk >> ./data/dev_clean.lst
grep -r -f ${selected_lists_path}/dev/clean.name /tmp/aishell_3 >> ./data/dev_clean.lst

#rir
for name_path in ${linear} ${circle} ${non_uniform}; do
    name=`basename ${name_path}`
    find ${name_path} -iname "*.wav" >/tmp/${name} 
    echo $name
    for mode in train dev ; do        
        grep -r -f ${selected_lists_path}/${mode}/${name}.name /tmp/${name} > ./data/${mode}_${name}_rir.lst
    done
done
