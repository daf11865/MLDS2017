#!/bin/bash

if [ ! -f "datum.tar.gz" ]
then
wget https://gitlab.com/dfke/MLDS2017_datum/raw/master/hw1/datum.tar.gz
fi

tar -zxvf datum.tar.gz

python main.py --testpath $1 --outpath $2
